struct PromptInterruptState {
    requested: AtomicBool,
    active_stream: Mutex<Option<TcpStream>>,
}

struct ActivePromptStream {
    state: Arc<PromptInterruptState>,
}

#[derive(Default)]
struct PromptLiveSession {
    messages: Vec<ChatTemplateMessage>,
    resident_tokens: Vec<i32>,
    stream: Option<TcpStream>,
    dirty: bool,
}

#[derive(Clone, Copy)]
struct PromptSessionReuseStats {
    outcome: &'static str,
    reused_tokens: usize,
    appended_prefill_tokens: usize,
    resident_tokens_before: usize,
    resident_tokens_after: usize,
}

impl Default for PromptSessionReuseStats {
    fn default() -> Self {
        Self {
            outcome: "disabled",
            reused_tokens: 0,
            appended_prefill_tokens: 0,
            resident_tokens_before: 0,
            resident_tokens_after: 0,
        }
    }
}

impl PromptLiveSession {
    fn mark_dirty(&mut self) {
        self.resident_tokens.clear();
        self.stream.take();
        self.dirty = true;
    }
}

impl PromptInterruptState {
    fn begin_request(&self) {
        self.requested.store(false, Ordering::SeqCst);
    }

    fn interrupt_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }

    fn take_interrupt(&self) -> bool {
        self.requested.swap(false, Ordering::SeqCst)
    }

    fn activate(self: &Arc<Self>, stream: &TcpStream) -> Result<ActivePromptStream> {
        let clone = stream
            .try_clone()
            .context("clone prompt stream for interrupt handling")?;
        *self
            .active_stream
            .lock()
            .map_err(|_| anyhow!("prompt interrupt stream lock poisoned"))? = Some(clone);
        Ok(ActivePromptStream {
            state: Arc::clone(self),
        })
    }

    fn clear_active_stream(&self) {
        if let Ok(mut active) = self.active_stream.lock() {
            *active = None;
        }
    }

    fn request_interrupt(&self) {
        let stream = self
            .active_stream
            .lock()
            .ok()
            .and_then(|active| active.as_ref().and_then(|stream| stream.try_clone().ok()));
        if let Some(stream) = stream {
            self.requested.store(true, Ordering::SeqCst);
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for ActivePromptStream {
    fn drop(&mut self) {
        self.state.clear_active_stream();
    }
}

fn install_prompt_interrupt_handler() -> Result<Arc<PromptInterruptState>> {
    static INTERRUPT_STATE: OnceLock<Arc<PromptInterruptState>> = OnceLock::new();
    if let Some(state) = INTERRUPT_STATE.get() {
        return Ok(Arc::clone(state));
    }

    let state = Arc::new(PromptInterruptState {
        requested: AtomicBool::new(false),
        active_stream: Mutex::new(None),
    });
    let handler_state = Arc::clone(&state);
    ctrlc::set_handler(move || {
        handler_state.request_interrupt();
    })
    .context("install Ctrl-C prompt interrupt handler")?;

    let _ = INTERRUPT_STATE.set(Arc::clone(&state));
    Ok(INTERRUPT_STATE.get().map(Arc::clone).unwrap_or(state))
}
