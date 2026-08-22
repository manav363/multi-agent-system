//! One agent's pane: what it is doing, and how it is doing.
//!
//! Every agent is on screen at once in an equal cell, so each pane has to say
//! everything about its agent in a small space — identity, health, and live
//! output — without becoming a wall of numbers.

use crate::core::agent::Agent;
use crate::core::events::AgentStatus;
use crate::metrics::tracker::AgentLiveMetrics;
use crate::tui::app::AgentView;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const DIM: Color = Color::Rgb(120, 122, 138);
const BODY: Color = Color::Rgb(214, 216, 226);
const RULE: Color = Color::Rgb(58, 62, 76);

/// Everything one pane needs.
pub struct PaneContext<'a> {
    pub agent: &'a Agent,
    pub view: &'a AgentView,
    pub metrics: Option<&'a AgentLiveMetrics>,
    pub focused: bool,
    pub spinner_idx: usize,
    /// False when the model server is unreachable — health is not just latency.
    pub provider_online: bool,
}

/// Status label and colour.
fn status_style(status: AgentStatus) -> (&'static str, Color) {
    match status {
        AgentStatus::Idle => ("IDLE", DIM),
        AgentStatus::Planning => ("PLANNING", Color::Cyan),
        AgentStatus::Thinking => ("THINKING", Color::Yellow),
        AgentStatus::Streaming => ("STREAMING", Color::LightGreen),
        AgentStatus::CallingTool => ("TOOL", Color::LightMagenta),
        AgentStatus::Evaluating => ("REVIEWING", Color::Magenta),
        AgentStatus::Done => ("DONE", Color::Green),
        AgentStatus::Error => ("ERROR", Color::Red),
    }
}

/// A single health indicator: connectivity, then latency, then throughput.
///
/// Connectivity is judged from evidence rather than assumed: an unreachable
/// server is down, an agent that has produced tokens is proven reachable, and
/// one that has not yet run is simply unknown.
fn health_dot(ctx: &PaneContext<'_>) -> (&'static str, Color, &'static str) {
    if !ctx.provider_online {
        return ("●", Color::Red, "offline");
    }
    if ctx.agent.status == AgentStatus::Error {
        return ("●", Color::Red, "failed");
    }
    match ctx.metrics.map(|m| m.total_tokens).unwrap_or(0) {
        0 => ("○", DIM, "idle"),
        _ => ("●", Color::Green, "ok"),
    }
}

/// `1.2s`, `340ms`, or a dash when there is nothing to report.
fn format_ms(ms: Option<u64>) -> String {
    match ms {
        Some(v) if v >= 1000 => format!("{:.1}s", v as f64 / 1000.0),
        Some(v) => format!("{v}ms"),
        None => "—".to_string(),
    }
}

/// Header: role icon, name, and live state.
fn header_line(ctx: &PaneContext<'_>) -> Line<'static> {
    let role = &ctx.agent.config.role;
    let (label, colour) = status_style(ctx.agent.status);
    let busy = matches!(
        ctx.agent.status,
        AgentStatus::Planning
            | AgentStatus::Thinking
            | AgentStatus::Streaming
            | AgentStatus::CallingTool
            | AgentStatus::Evaluating
    );

    let mut spans = vec![
        Span::styled(
            format!("{} ", role.icon()),
            Style::default().fg(role.default_color()),
        ),
        Span::styled(
            ctx.agent.config.name.clone(),
            Style::default()
                .fg(role.default_color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
    ];

    if busy {
        spans.push(Span::styled(
            format!("{} ", SPINNER[ctx.spinner_idx % SPINNER.len()]),
            Style::default().fg(colour),
        ));
    }
    spans.push(Span::styled(
        label,
        Style::default().fg(colour).add_modifier(Modifier::BOLD),
    ));

    if let Some(secs) = ctx.view.elapsed_secs() {
        spans.push(Span::styled(
            format!(" {secs:.0}s"),
            Style::default().fg(DIM),
        ));
    } else if let Some(ms) = ctx.view.last_duration_ms {
        spans.push(Span::styled(
            format!(" {}", format_ms(Some(ms))),
            Style::default().fg(DIM),
        ));
    }

    Line::from(spans)
}

/// Health: connectivity, model, latency, throughput, volume.
fn health_line(ctx: &PaneContext<'_>) -> Line<'static> {
    let (dot, dot_colour, dot_label) = health_dot(ctx);
    let ttft = ctx.metrics.and_then(|m| m.ttft_ms);
    let tps = ctx.metrics.map(|m| m.avg_tps).unwrap_or(0.0);
    let tokens = ctx.metrics.map(|m| m.total_tokens).unwrap_or(0);

    Line::from(vec![
        Span::styled(format!("{dot} "), Style::default().fg(dot_colour)),
        Span::styled(dot_label, Style::default().fg(dot_colour)),
        Span::styled(" · ", Style::default().fg(RULE)),
        Span::styled(
            ctx.agent.config.model.clone(),
            Style::default().fg(Color::Rgb(150, 160, 190)),
        ),
        Span::styled(" · ", Style::default().fg(RULE)),
        Span::styled("ttft ", Style::default().fg(DIM)),
        Span::styled(format_ms(ttft), Style::default().fg(Color::Cyan)),
        Span::styled(" · ", Style::default().fg(RULE)),
        Span::styled(
            format!("{tps:.0} tok/s"),
            Style::default().fg(if tps > 0.0 { Color::LightGreen } else { DIM }),
        ),
        Span::styled(" · ", Style::default().fg(RULE)),
        Span::styled(format!("{tokens} tok"), Style::default().fg(DIM)),
    ])
}

/// Tool activity, reasoning volume, or an error — whichever is worth the row.
fn activity_line(ctx: &PaneContext<'_>) -> Option<Line<'static>> {
    if let Some(err) = &ctx.view.error {
        return Some(Line::from(vec![
            Span::styled("✗ ", Style::default().fg(Color::Red)),
            Span::styled(err.clone(), Style::default().fg(Color::Rgb(210, 140, 140))),
        ]));
    }

    if let Some(tool) = &ctx.view.last_tool {
        let (mark, colour) = if tool.running {
            (SPINNER[ctx.spinner_idx % SPINNER.len()], Color::Yellow)
        } else if tool.is_error {
            ("✗", Color::Red)
        } else {
            ("✓", Color::LightGreen)
        };
        let timing = if tool.running {
            String::new()
        } else {
            format!(" {}", format_ms(Some(tool.duration_ms)))
        };
        return Some(Line::from(vec![
            Span::styled(format!("{mark} "), Style::default().fg(colour)),
            Span::styled(
                tool.name.clone(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(timing, Style::default().fg(DIM)),
            Span::styled(
                if ctx.view.tool_calls > 1 {
                    format!("  ({} calls)", ctx.view.tool_calls)
                } else {
                    String::new()
                },
                Style::default().fg(DIM),
            ),
        ]));
    }

    (ctx.view.thought_lines > 0).then(|| {
        Line::from(Span::styled(
            format!("💭 {} lines of reasoning", ctx.view.thought_lines),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ))
    })
}

/// Wrap the agent's output into body lines, tailing the newest content.
fn body_lines(text: &str, width: u16, height: u16, placeholder: &str) -> Vec<Line<'static>> {
    if text.trim().is_empty() {
        return vec![Line::from(Span::styled(
            placeholder.to_string(),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ))];
    }

    let wrap_at = width.max(1) as usize;
    let mut wrapped: Vec<String> = Vec::new();
    for raw in text.lines() {
        if raw.is_empty() {
            wrapped.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in raw.split_inclusive(' ') {
            if current.chars().count() + word.chars().count() > wrap_at && !current.is_empty() {
                wrapped.push(std::mem::take(&mut current));
            }
            current.push_str(word);
        }
        if !current.is_empty() {
            wrapped.push(current);
        }
    }

    // A live pane follows the newest output; there is no room to scroll back.
    let start = wrapped.len().saturating_sub(height.max(1) as usize);
    wrapped[start..]
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(BODY))))
        .collect()
}

/// Render one agent pane into `area`.
pub fn render_agent_pane(f: &mut Frame, area: Rect, ctx: &PaneContext<'_>) {
    let role_colour = ctx.agent.config.role.default_color();
    let busy = !matches!(ctx.agent.status, AgentStatus::Idle | AgentStatus::Done);

    let border_colour = if ctx.focused {
        Color::White
    } else if busy {
        role_colour
    } else {
        RULE
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(if ctx.focused {
            BorderType::Double
        } else {
            BorderType::Plain
        })
        .border_style(Style::default().fg(border_colour))
        .title(Span::styled(
            format!(" {} ", ctx.agent.config.role.name()),
            Style::default()
                .fg(role_colour)
                .add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 3 {
        return;
    }

    let activity = activity_line(ctx);
    let chrome = if activity.is_some() { 3 } else { 2 };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(chrome), Constraint::Min(1)])
        .split(inner);

    let mut head = vec![header_line(ctx), health_line(ctx)];
    if let Some(activity) = activity {
        head.push(activity);
    }
    f.render_widget(Paragraph::new(head), rows[0]);

    let placeholder = match ctx.agent.status {
        AgentStatus::Idle => "waiting for its turn",
        AgentStatus::Error => "step failed",
        _ => "working…",
    };
    f.render_widget(
        Paragraph::new(body_lines(
            &ctx.view.output,
            rows[1].width,
            rows[1].height,
            placeholder,
        ))
        .wrap(Wrap { trim: false }),
        rows[1],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_read_naturally_at_both_scales() {
        assert_eq!(format_ms(Some(340)), "340ms");
        assert_eq!(format_ms(Some(1500)), "1.5s");
        assert_eq!(format_ms(None), "—");
    }

    #[test]
    fn body_tails_the_newest_output_when_it_overflows() {
        let text = (1..=40).map(|i| format!("line {i}\n")).collect::<String>();
        let lines = body_lines(&text, 40, 5, "idle");
        assert_eq!(lines.len(), 5);
        let last = lines.last().unwrap().spans[0].content.clone();
        assert!(
            last.contains("line 40"),
            "should show the newest, got {last}"
        );
    }

    #[test]
    fn body_wraps_long_lines_to_the_pane_width() {
        let text = "word ".repeat(60);
        let lines = body_lines(&text, 20, 40, "idle");
        assert!(lines.len() > 1, "a 300-char line must wrap in 20 columns");
        assert!(lines
            .iter()
            .all(|l| l.spans[0].content.chars().count() <= 21));
    }

    #[test]
    fn an_empty_pane_says_what_it_is_waiting_for() {
        let lines = body_lines("", 40, 5, "waiting for its turn");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].spans[0].content.contains("waiting"));
    }

    #[test]
    fn multibyte_output_wraps_without_panicking() {
        let text = "日本語のテキスト🛡️ ".repeat(30);
        let lines = body_lines(&text, 12, 6, "idle");
        assert!(!lines.is_empty());
    }

    #[test]
    fn every_status_has_a_distinct_label() {
        let all = [
            AgentStatus::Idle,
            AgentStatus::Planning,
            AgentStatus::Thinking,
            AgentStatus::Streaming,
            AgentStatus::CallingTool,
            AgentStatus::Evaluating,
            AgentStatus::Done,
            AgentStatus::Error,
        ];
        let mut labels: Vec<&str> = all.iter().map(|s| status_style(*s).0).collect();
        labels.sort_unstable();
        let before = labels.len();
        labels.dedup();
        assert_eq!(before, labels.len(), "two states share a label");
    }
}
