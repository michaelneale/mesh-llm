use iroh::{Endpoint, EndpointAddr, TransportAddr};
use serde::Serialize;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct DirectConnectivityStatus {
    pub(crate) direct_candidates: Vec<String>,
    pub(crate) direct_candidate_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) public_ipv4_candidate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) invite_public_candidate: Option<String>,
    pub(crate) iroh_portmapper: IrohPortmapperStatus,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct IrohPortmapperStatus {
    pub(crate) enabled: bool,
    pub(crate) mapping_attempts: u64,
    pub(crate) external_address_updates: u64,
    pub(crate) has_reported_external_address: bool,
}

/// Return the first public IPv4 candidate iroh already discovered.
///
/// iroh relays provide address discovery. When a fixed bind port is supplied,
/// pair the iroh-discovered public IP with that port for user-managed forwarding.
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

pub(crate) fn iroh_status(
    endpoint: &Endpoint,
    invite_public_candidate: Option<SocketAddr>,
) -> DirectConnectivityStatus {
    let endpoint_addr = endpoint.addr();
    let direct_candidates: Vec<SocketAddr> = endpoint_addr
        .addrs
        .iter()
        .filter_map(|addr| match addr {
            TransportAddr::Ip(sock) => Some(*sock),
            TransportAddr::Relay(_) => None,
            _ => None,
        })
        .collect();
    let public_ipv4_candidate = direct_candidates.iter().find_map(|sock| match sock.ip() {
        std::net::IpAddr::V4(v4) if is_public_ipv4_addr(v4) => Some(sock.to_string()),
        _ => None,
    });
    let metrics = endpoint.metrics();
    let mapping_attempts = metrics.net_report.portmap_attempts.get();
    let external_address_updates = metrics.net_report.portmap_external_address_updated.get();

    DirectConnectivityStatus {
        direct_candidate_count: direct_candidates.len(),
        direct_candidates: direct_candidates
            .into_iter()
            .map(|addr| addr.to_string())
            .collect(),
        public_ipv4_candidate,
        invite_public_candidate: invite_public_candidate.map(|addr| addr.to_string()),
        iroh_portmapper: IrohPortmapperStatus {
            enabled: true,
            mapping_attempts,
            external_address_updates,
            has_reported_external_address: external_address_updates > 0,
        },
    }
}

pub(super) fn emit_iroh_status(endpoint: &Endpoint, invite_public_candidate: Option<SocketAddr>) {
    let status = iroh_status(endpoint, invite_public_candidate);
    let candidate = status
        .invite_public_candidate
        .as_deref()
        .or(status.public_ipv4_candidate.as_deref())
        .unwrap_or("none");
    let _ = crate::cli::output::emit_event(crate::cli::output::OutputEvent::Info {
        message: format!(
            "iroh direct connectivity: {} direct candidate(s), public candidate {candidate}, portmapper enabled (attempts={}, external_updates={})",
            status.direct_candidate_count,
            status.iroh_portmapper.mapping_attempts,
            status.iroh_portmapper.external_address_updates
        ),
        context: None,
    });
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
        "Direct UDP port forwarding may not work on --bind-port {bind_port}: router WAN appears to be {nat_kind} while iroh discovered a different public IPv4. Router port forwards only cover the local router; ask the ISP to opt out of CGNAT/provide a public IPv4, bridge the upstream router, or use relay/tunnel mode."
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
