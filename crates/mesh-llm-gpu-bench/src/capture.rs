use anyhow::{anyhow, Context, Result};
use std::ffi::c_int;
use std::io::Read;

pub fn capture_stdout(call: unsafe extern "C" fn() -> c_int) -> Result<Vec<u8>> {
    let mut pipe_fds = [0; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("failed to create stdout pipe");
    }

    let stdout_fd = unsafe { libc::dup(libc::STDOUT_FILENO) };
    if stdout_fd < 0 {
        unsafe {
            libc::close(pipe_fds[0]);
            libc::close(pipe_fds[1]);
        }
        return Err(std::io::Error::last_os_error()).context("failed to duplicate stdout");
    }

    if unsafe { libc::dup2(pipe_fds[1], libc::STDOUT_FILENO) } < 0 {
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
        libc::dup2(stdout_fd, libc::STDOUT_FILENO);
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

#[cfg(unix)]
use std::os::fd::FromRawFd;
