/// Collect only the requested hardware metrics.
pub fn query(metrics: &[Metric]) -> HardwareSurvey {
    let collector = detect_collector();
    let mut survey = collector.collect(metrics);
    if metrics.contains(&Metric::Hostname) {
        survey.hostname = detect_hostname();
    }
    if metrics.contains(&Metric::GpuFacts) {
        hydrate_gpu_facts(&mut survey, metrics);
    }
    survey
}

pub fn survey() -> HardwareSurvey {
    query(&[
        Metric::GpuName,
        Metric::VramBytes,
        Metric::GpuCount,
        Metric::Hostname,
        Metric::IsSoc,
        Metric::GpuFacts,
    ])
}
