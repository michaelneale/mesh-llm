use anyhow::{anyhow, Context, Result};
use std::ffi::c_int;

const STDOUT_FD: c_int = 1;

#[cfg(unix)]
use std::io::Read;

#[cfg(unix)]
pub fn capture_stdout(call: unsafe extern "C" fn() -> c_int) -> Result<Vec<u8>> {
    let mut pipe_fds = [0; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("failed to create stdout pipe");
    }

    let stdout_fd = unsafe { libc::dup(STDOUT_FD) };
    if stdout_fd < 0 {
        unsafe {
            libc::close(pipe_fds[0]);
            libc::close(pipe_fds[1]);
        }
        return Err(std::io::Error::last_os_error()).context("failed to duplicate stdout");
    }

    if unsafe { libc::dup2(pipe_fds[1], STDOUT_FD) } < 0 {
        unsafe {
            libc::close(stdout_fd);
            libc::close(pipe_fds[0]);
            libc::close(pipe_fds[1]);
        }
        return Err(std::io::Error::last_os_error()).context("failed to redirect stdout");
    }

    let status = unsafe { call() };
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::dup2(stdout_fd, STDOUT_FD);
        libc::close(stdout_fd);
        libc::close(pipe_fds[1]);
    }

    let mut output = Vec::new();
    let mut reader = unsafe { std::fs::File::from_raw_fd(pipe_fds[0]) };
    reader
        .read_to_end(&mut output)
        .context("failed to read captured benchmark output")?;

    if status != 0 {
        return Err(anyhow!(
            "native benchmark backend exited with status {status}"
        ));
    }

    Ok(output)
}

#[cfg(windows)]
pub fn capture_stdout(call: unsafe extern "C" fn() -> c_int) -> Result<Vec<u8>> {
    let mut pipe_fds = [0; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr(), 64 * 1024, libc::O_BINARY) } != 0 {
        return Err(std::io::Error::last_os_error()).context("failed to create stdout pipe");
    }

    let stdout_fd = unsafe { libc::dup(STDOUT_FD) };
    if stdout_fd < 0 {
        unsafe {
            libc::close(pipe_fds[0]);
            libc::close(pipe_fds[1]);
        }
        return Err(std::io::Error::last_os_error()).context("failed to duplicate stdout");
    }

    if unsafe { libc::dup2(pipe_fds[1], STDOUT_FD) } < 0 {
        unsafe {
            libc::close(stdout_fd);
            libc::close(pipe_fds[0]);
            libc::close(pipe_fds[1]);
        }
        return Err(std::io::Error::last_os_error()).context("failed to redirect stdout");
    }

    let status = unsafe { call() };
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::dup2(stdout_fd, STDOUT_FD);
        libc::close(stdout_fd);
        libc::close(pipe_fds[1]);
    }

    let output = read_pipe_to_end(pipe_fds[0])?;

    if status != 0 {
        return Err(anyhow!(
            "native benchmark backend exited with status {status}"
        ));
    }

    Ok(output)
}

#[cfg(windows)]
fn read_pipe_to_end(fd: c_int) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = unsafe {
            libc::read(
                fd,
                buffer.as_mut_ptr().cast::<std::ffi::c_void>(),
                buffer.len() as libc::c_uint,
            )
        };
        if bytes_read > 0 {
            output.extend_from_slice(&buffer[..bytes_read as usize]);
            continue;
        }
        if bytes_read == 0 {
            unsafe {
                libc::close(fd);
            }
            return Ok(output);
        }

        let err = std::io::Error::last_os_error();
        unsafe {
            libc::close(fd);
        }
        return Err(err).context("failed to read captured benchmark output");
    }
}

#[cfg(unix)]
use std::os::fd::FromRawFd;
