use iroh::{EndpointAddr, TransportAddr};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

/// Return the first public IPv4 candidate iroh already discovered.
///
/// iroh relays provide address discovery, so prefer their endpoint candidates
/// before falling back to external STUN. When a fixed bind port is supplied,
/// pair the discovered public IP with that port for user-managed forwarding.
pub(super) fn iroh_public_addr(addr: &EndpointAddr, advertised_port: u16) -> Option<SocketAddr> {
    addr.addrs
        .iter()
        .find_map(|transport_addr| match transport_addr {
            TransportAddr::Ip(sock) => match sock.ip() {
                std::net::IpAddr::V4(v4) if is_public_ipv4_addr(v4) => {
                    Some(SocketAddr::V4(SocketAddrV4::new(v4, advertised_port)))
                }
                _ => None,
            },
            _ => None,
        })
}

/// Discover our public IP via external STUN only when iroh did not provide it.
///
/// We can't send STUN from the bound port (iroh owns it), but we only need
/// the public IP; the port is known from --bind-port plus router forwarding.
pub(super) async fn fallback_stun_public_addr(advertised_port: u16) -> Option<SocketAddr> {
    let stun_servers = [
        "stun.l.google.com:19302",
        "stun.cloudflare.com:3478",
        "stun.stunprotocol.org:3478",
    ];

    // Bind to an ephemeral port; we only care about the IP, not the mapped port.
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;

    for server in &stun_servers {
        // STUN Binding Request: type=0x0001, len=0, magic=0x2112A442, txn=random.
        let mut req = [0u8; 20];
        req[0] = 0x00;
        req[1] = 0x01;
        req[4] = 0x21;
        req[5] = 0x12;
        req[6] = 0xA4;
        req[7] = 0x42;
        rand::fill(&mut req[8..20]);

        let addrs = match tokio::net::lookup_host(server).await {
            Ok(addrs) => addrs,
            Err(_) => continue,
        };

        for dest in addrs.filter(SocketAddr::is_ipv4) {
            if sock.send_to(&req, dest).await.is_err() {
                continue;
            }

            let mut buf = [0u8; 256];
            let len = match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                sock.recv_from(&mut buf),
            )
            .await
            {
                Ok(Ok((len, _))) if len >= 20 => len,
                _ => continue,
            };

            if let Some(addr) = parse_stun_mapped_ipv4(&req, &buf[..len], advertised_port) {
                tracing::info!("STUN discovered public address: {addr}");
                return Some(addr);
            }
        }
    }

    tracing::warn!("STUN: could not discover public address");
    None
}

fn parse_stun_mapped_ipv4(
    request: &[u8; 20],
    response: &[u8],
    advertised_port: u16,
) -> Option<SocketAddr> {
    let magic = &request[4..8];
    let mut i = 20;
    while i + 4 <= response.len() {
        let attr_type = u16::from_be_bytes([response[i], response[i + 1]]);
        let attr_len = u16::from_be_bytes([response[i + 2], response[i + 3]]) as usize;
        if i + 4 + attr_len > response.len() {
            break;
        }
        let val = &response[i + 4..i + 4 + attr_len];

        if attr_type == 0x0020 && attr_len >= 8 && val[1] == 0x01 {
            let ip = Ipv4Addr::new(
                val[4] ^ magic[0],
                val[5] ^ magic[1],
                val[6] ^ magic[2],
                val[7] ^ magic[3],
            );
            return Some(SocketAddr::V4(SocketAddrV4::new(ip, advertised_port)));
        }
        if attr_type == 0x0001 && attr_len >= 8 && val[1] == 0x01 {
            let ip = Ipv4Addr::new(val[4], val[5], val[6], val[7]);
            return Some(SocketAddr::V4(SocketAddrV4::new(ip, advertised_port)));
        }

        i += (4 + (attr_len + 3)) & !3;
    }
    None
}

pub(super) fn is_cgnat_ipv4_addr(addr: Ipv4Addr) -> bool {
    let octets = addr.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

pub(super) fn is_public_ipv4_addr(addr: Ipv4Addr) -> bool {
    !addr.is_private()
        && !addr.is_loopback()
        && !addr.is_link_local()
        && !addr.is_multicast()
        && !addr.is_broadcast()
        && !addr.is_documentation()
        && !addr.is_unspecified()
        && !is_cgnat_ipv4_addr(addr)
}

async fn upnp_router_wan_ipv4() -> Option<Ipv4Addr> {
    let location = upnp_igd_location().await?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()?;
    let description = client
        .get(location.clone())
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;
    let control_url = upnp_wan_control_url(&location, &description)?;
    let service_type = upnp_wan_service_type(&description)?;
    let body = format!(
        r#"<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
<s:Body><u:GetExternalIPAddress xmlns:u="{service_type}"></u:GetExternalIPAddress></s:Body>
</s:Envelope>"#
    );
    let response = client
        .post(control_url)
        .header("content-type", r#"text/xml; charset="utf-8""#)
        .header(
            "soapaction",
            format!(r#""{service_type}#GetExternalIPAddress""#),
        )
        .body(body)
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;
    xml_tag_text(&response, "NewExternalIPAddress")?
        .parse()
        .ok()
}

async fn upnp_igd_location() -> Option<String> {
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.ok()?;
    let request = concat!(
        "M-SEARCH * HTTP/1.1\r\n",
        "HOST: 239.255.255.250:1900\r\n",
        "MAN: \"ssdp:discover\"\r\n",
        "MX: 2\r\n",
        "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n",
        "\r\n"
    );
    sock.send_to(request.as_bytes(), "239.255.255.250:1900")
        .await
        .ok()?;

    let mut buf = [0u8; 2048];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        let remaining = deadline.checked_duration_since(std::time::Instant::now())?;
        let (len, _) = tokio::time::timeout(remaining, sock.recv_from(&mut buf))
            .await
            .ok()?
            .ok()?;
        let response = std::str::from_utf8(&buf[..len]).ok()?;
        if let Some(location) = http_header_value(response, "location") {
            return Some(location);
        }
    }
}

fn http_header_value(response: &str, name: &str) -> Option<String> {
    response.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

fn upnp_wan_control_url(location: &str, description: &str) -> Option<String> {
    let block = upnp_wan_service_block(description)?;
    let control_url = xml_tag_text(block, "controlURL")?;
    let base = url::Url::parse(location).ok()?;
    base.join(&control_url).ok().map(|url| url.to_string())
}

fn upnp_wan_service_type(description: &str) -> Option<String> {
    let block = upnp_wan_service_block(description)?;
    xml_tag_text(block, "serviceType")
}

fn upnp_wan_service_block(description: &str) -> Option<&str> {
    description.split("</service>").find(|block| {
        block.contains("urn:schemas-upnp-org:service:WANIPConnection:")
            || block.contains("urn:schemas-upnp-org:service:WANPPPConnection:")
    })
}

fn xml_tag_text(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = xml.find(&start_tag)? + start_tag.len();
    let end = xml[start..].find(&end_tag)? + start;
    Some(xml[start..end].trim().to_string())
}

pub(super) async fn warn_if_cgnat_likely(bind_port: Option<u16>, public_addr: Option<SocketAddr>) {
    let Some(bind_port) = bind_port else {
        return;
    };
    let Some(SocketAddr::V4(public_addr)) = public_addr else {
        return;
    };
    let public_ip = *public_addr.ip();
    if !is_public_ipv4_addr(public_ip) {
        return;
    }
    let Some(router_wan_ip) = upnp_router_wan_ipv4().await else {
        return;
    };
    if router_wan_ip == public_ip || is_public_ipv4_addr(router_wan_ip) {
        return;
    }

    let nat_kind = if is_cgnat_ipv4_addr(router_wan_ip) {
        "CGNAT"
    } else {
        "private/double NAT"
    };
    emit_warning(format!(
        "Direct UDP port forwarding may not work on --bind-port {bind_port}: router WAN appears to be {nat_kind} while STUN sees a different public IPv4. Router port forwards only cover the local router; ask the ISP to opt out of CGNAT/provide a public IPv4, bridge the upstream router, or use relay/tunnel mode."
    ));
}

fn emit_warning(message: String) {
    let _ = crate::cli::output::emit_event(crate::cli::output::OutputEvent::Warning {
        message,
        context: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_ipv4_classification_excludes_cgnat_and_private_ranges() {
        assert!(is_public_ipv4_addr("8.8.8.8".parse().unwrap()));
        assert!(!is_public_ipv4_addr("100.64.0.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("100.127.255.254".parse().unwrap()));
        assert!(!is_public_ipv4_addr("10.0.0.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("172.16.0.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("192.168.1.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("169.254.1.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("127.0.0.1".parse().unwrap()));
        assert!(!is_public_ipv4_addr("192.0.2.10".parse().unwrap()));
    }

    #[test]
    fn upnp_wan_helpers_extract_wan_service_endpoint() {
        let description = r#"
            <root>
              <device>
                <serviceList>
                  <service>
                    <serviceType>urn:schemas-upnp-org:service:Layer3Forwarding:1</serviceType>
                    <controlURL>/ctl/L3F</controlURL>
                  </service>
                  <service>
                    <serviceType>urn:schemas-upnp-org:service:WANIPConnection:2</serviceType>
                    <controlURL>/ctl/IPConn</controlURL>
                  </service>
                </serviceList>
              </device>
            </root>
        "#;

        assert_eq!(
            upnp_wan_service_type(description).as_deref(),
            Some("urn:schemas-upnp-org:service:WANIPConnection:2")
        );
        assert_eq!(
            upnp_wan_control_url("http://192.168.1.1:5000/rootDesc.xml", description).as_deref(),
            Some("http://192.168.1.1:5000/ctl/IPConn")
        );
    }

    #[test]
    fn iroh_public_addr_prefers_endpoint_candidates() {
        let mut addrs = std::collections::BTreeSet::new();
        addrs.insert(TransportAddr::Ip("100.72.12.34:9999".parse().unwrap()));
        addrs.insert(TransportAddr::Ip("198.51.100.10:9999".parse().unwrap()));
        addrs.insert(TransportAddr::Ip("203.0.113.10:9999".parse().unwrap()));
        addrs.insert(TransportAddr::Ip("8.8.8.8:9999".parse().unwrap()));
        let addr = EndpointAddr {
            id: iroh::SecretKey::generate().public(),
            addrs,
        };

        assert_eq!(
            iroh_public_addr(&addr, 53238),
            Some("8.8.8.8:53238".parse().unwrap())
        );
    }
}
