use super::formatters::{plan_json, print_json, JsonFormatter, MoePlanFormatter};
use crate::system::moe_planner::MoePlanReport;
use anyhow::Result;

impl MoePlanFormatter for JsonFormatter {
    fn render(&self, report: &MoePlanReport) -> Result<()> {
        print_json(plan_json(report))
    }
}
