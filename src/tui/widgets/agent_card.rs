use crate::core::agent::Agent;
use crate::core::events::AgentStatus;
use crate::metrics::tracker::AgentLiveMetrics;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn render_agent_pipeline_cards(
    f: &mut Frame,
    area: Rect,
    agents: &[&Agent],
    metrics: &std::collections::HashMap<String, AgentLiveMetrics>,
    spinner_idx: usize,
    selected_agent_idx: Option<usize>,
) {
    if agents.is_empty() {
        return;
    }

    let constraints = vec![Constraint::Ratio(1, agents.len() as u32); agents.len()];
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (idx, agent) in agents.iter().enumerate() {
        if idx >= chunks.len() {
            break;
        }

        let is_selected = selected_agent_idx == Some(idx);
        let agent_metrics = metrics.get(&agent.config.id);

        let (status_icon, status_color) = match agent.status {
            AgentStatus::Idle => ("💤 IDLE", Color::DarkGray),
            AgentStatus::Planning => (SPINNER_FRAMES[spinner_idx % SPINNER_FRAMES.len()], Color::Cyan),
            AgentStatus::Thinking => (SPINNER_FRAMES[spinner_idx % SPINNER_FRAMES.len()], Color::Yellow),
            AgentStatus::Streaming => ("⚡ STREAM", Color::Green),
            AgentStatus::CallingTool => ("🛠️ TOOL", Color::LightMagenta),
            AgentStatus::Evaluating => ("🔍 CRITIQUE", Color::Magenta),
            AgentStatus::Done => ("✓ DONE", Color::LightGreen),
            AgentStatus::Error => ("✗ ERR", Color::Red),
        };

        let border_color = if is_selected {
            Color::LightCyan
        } else if agent.status == AgentStatus::Streaming || agent.status == AgentStatus::CallingTool {
            status_color
        } else {
            Color::Rgb(60, 65, 80)
        };

        let tps = agent_metrics.map(|m| m.current_tps).unwrap_or(0.0);
        let tokens = agent_metrics.map(|m| m.total_tokens).unwrap_or(0);
        let ttft = agent_metrics.and_then(|m| m.ttft_ms).map(|ms| format!("{}ms", ms)).unwrap_or_else(|| "-".to_string());

        let lines = vec![
            Line::from(vec![
                Span::styled(format!("{} ", agent.config.role.icon()), Style::default()),
                Span::styled(&agent.config.name, Style::default().add_modifier(Modifier::BOLD).fg(agent.config.role.default_color())),
            ]),
            Line::from(vec![
                Span::styled(format!("M: {}", agent.config.model), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("State: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{} {}", status_icon, agent.status.as_str()), Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled(format!("Tokens: {:<4} | TTFT: {:<5}", tokens, ttft), Style::default().fg(Color::Rgb(150, 150, 160))),
            ]),
            Line::from(vec![
                Span::styled(format!("Speed: {:>4.1} tok/s", tps), Style::default().fg(if tps > 0.0 { Color::LightGreen } else { Color::DarkGray })),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled(
                format!(" {} ", agent.config.role.name()),
                Style::default().fg(agent.config.role.default_color()).add_modifier(Modifier::BOLD),
            ));

        let paragraph = Paragraph::new(lines)
            .block(block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, chunks[idx]);
    }
}
