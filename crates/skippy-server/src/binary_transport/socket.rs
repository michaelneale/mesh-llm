use super::*;

#[cfg(target_os = "macos")]
use std::{net::Ipv4Addr, os::fd::AsRawFd, ptr};

pub(super) fn downstream_source_ip(config: &StageConfig) -> Result<Option<IpAddr>> {
    let bind_addr = config
        .bind_addr
        .parse::<SocketAddr>()
        .with_context(|| format!("parse stage bind_addr {}", config.bind_addr))?;
    let ip = bind_addr.ip();
    if ip.is_unspecified() {
        Ok(None)
    } else {
        Ok(Some(ip))
    }
}

pub(super) fn resolve_downstream_endpoint(endpoint: &str) -> Result<SocketAddr> {
    endpoint
        .to_socket_addrs()
        .with_context(|| format!("resolve downstream binary stage endpoint {endpoint}"))?
        .find(SocketAddr::is_ipv4)
        .or_else(|| endpoint.to_socket_addrs().ok()?.next())
        .with_context(|| {
            format!("downstream binary stage endpoint resolved no addresses: {endpoint}")
        })
}

pub(super) fn connect_downstream_socket(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let mut errors = Vec::new();

    macro_rules! try_connect {
        ($mode:literal, $connect:expr) => {
            match $connect {
                Ok(stream) => return Ok(stream),
                Err(error) => {
                    eprintln!(
                        "downstream connect retry: source={source_ip:?} remote={downstream_addr} mode={} error={error}",
                        $mode
                    );
                    errors.push(format!("{} failed: {error}", $mode));
                }
            }
        };
    }

    try_connect!(
        "route-selected",
        connect_route_selected_with_timeout(downstream_addr, source_ip, timeout)
    );
    try_connect!(
        "bound-interface",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, true)
    );
    try_connect!(
        "source-bound",
        connect_bound_with_timeout(downstream_addr, source_ip, timeout, false)
    );
    try_connect!(
        "blocking-source-bound",
        connect_blocking_with_timeout(downstream_addr, source_ip, timeout, false)
    );
    try_connect!(
        "blocking-route-selected",
        connect_route_selected_blocking_with_timeout(downstream_addr, source_ip, timeout)
    );

    Err(io::Error::new(io::ErrorKind::Other, errors.join("; ")))
}

pub(super) fn connect_route_selected_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&downstream_addr, timeout)?;
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect succeeded: source={source_ip:?} remote={downstream_addr} mode=route-selected"
    );
    Ok(stream)
}

pub(super) fn connect_route_selected_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = TcpStream::connect(downstream_addr)
            .and_then(|stream| validate_route_selected_stream(stream, source_ip));
        let _ = tx.send(result);
    });
    rx.recv_timeout(timeout).map_err(|_| {
        io::Error::new(
            io::ErrorKind::TimedOut,
            "blocking route-selected fallback connect timed out",
        )
    })?
}

pub(super) fn validate_route_selected_stream(
    stream: TcpStream,
    source_ip: Option<IpAddr>,
) -> io::Result<TcpStream> {
    if let Some(source_ip) = source_ip {
        let local_ip = stream.local_addr()?.ip();
        if local_ip != source_ip {
            return Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                format!("route-selected local address {local_ip} did not match {source_ip}"),
            ));
        }
    }
    eprintln!(
        "downstream connect retry succeeded: source={source_ip:?} mode=blocking-route-selected"
    );
    Ok(stream)
}

pub(super) fn connect_bound_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect_timeout(&downstream_addr, timeout);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect_timeout(&SockAddr::from(downstream_addr), timeout)?;
    Ok(socket.into())
}

pub(super) fn connect_blocking_with_timeout(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    timeout: Duration,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = tx.send(connect_bound_blocking(
            downstream_addr,
            source_ip,
            bind_interface,
        ));
    });
    rx.recv_timeout(timeout)
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "fallback connect timed out"))?
}

pub(super) fn connect_bound_blocking(
    downstream_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    bind_interface: bool,
) -> io::Result<TcpStream> {
    let Some(source_ip) = source_ip else {
        return TcpStream::connect(downstream_addr);
    };
    let domain = match (source_ip, downstream_addr) {
        (IpAddr::V4(_), SocketAddr::V4(_)) => Domain::IPV4,
        (IpAddr::V6(_), SocketAddr::V6(_)) => Domain::IPV6,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("source address {source_ip} cannot connect to {downstream_addr}"),
            ));
        }
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.bind(&SockAddr::from(SocketAddr::new(source_ip, 0)))?;
    if bind_interface {
        bind_socket_to_source_interface(&socket, source_ip)?;
    }
    socket.connect(&SockAddr::from(downstream_addr))?;
    Ok(socket.into())
}

#[cfg(target_os = "macos")]
pub(super) fn bind_socket_to_source_interface(
    socket: &Socket,
    source_ip: IpAddr,
) -> io::Result<()> {
    let Some(interface_index) = interface_index_for_ip(source_ip)? else {
        return Ok(());
    };
    let interface_index = interface_index as libc::c_int;
    let (level, optname) = match source_ip {
        IpAddr::V4(_) => (libc::IPPROTO_IP, 25),
        IpAddr::V6(_) => (libc::IPPROTO_IPV6, 125),
    };
    let result = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            level,
            optname,
            ptr::addr_of!(interface_index).cast(),
            std::mem::size_of_val(&interface_index) as libc::socklen_t,
        )
    };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(super) fn bind_socket_to_source_interface(
    _socket: &Socket,
    _source_ip: IpAddr,
) -> io::Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn interface_index_for_ip(source_ip: IpAddr) -> io::Result<Option<u32>> {
    let mut addrs: *mut libc::ifaddrs = ptr::null_mut();
    let result = unsafe { libc::getifaddrs(ptr::addr_of_mut!(addrs)) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    let mut cursor = addrs;
    while !cursor.is_null() {
        let ifaddr = unsafe { &*cursor };
        if !ifaddr.ifa_addr.is_null() && sockaddr_ip(ifaddr.ifa_addr) == Some(source_ip) {
            let index = unsafe { libc::if_nametoindex(ifaddr.ifa_name) };
            unsafe { libc::freeifaddrs(addrs) };
            if index == 0 {
                return Err(io::Error::last_os_error());
            }
            return Ok(Some(index));
        }
        cursor = ifaddr.ifa_next;
    }
    unsafe { libc::freeifaddrs(addrs) };
    Ok(None)
}

#[cfg(target_os = "macos")]
pub(super) fn sockaddr_ip(addr: *const libc::sockaddr) -> Option<IpAddr> {
    match unsafe { (*addr).sa_family as libc::c_int } {
        libc::AF_INET => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in>()) };
            Some(IpAddr::V4(Ipv4Addr::from(
                addr.sin_addr.s_addr.to_ne_bytes(),
            )))
        }
        libc::AF_INET6 => {
            let addr = unsafe { &*(addr.cast::<libc::sockaddr_in6>()) };
            Some(IpAddr::V6(addr.sin6_addr.s6_addr.into()))
        }
        _ => None,
    }
}
