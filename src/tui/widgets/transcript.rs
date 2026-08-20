use crate::core::agent::AgentRole;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

#[derive(Debug, Clone)]
pub enum TranscriptItem {
    UserGoal {
        text: String,
        timestamp: String,
    },
    AgentOutput {
        agent_id: String,
        agent_name: String,
        role: AgentRole,
        text: String,
        thoughts: Option<String>,
        is_streaming: bool,
    },
    ToolExecution {
        agent_name: String,
        tool_name: String,
        args: String,
        output: String,
        is_error: bool,
        duration_ms: u64,
    },
    Milestone {
        step_title: String,
        duration_ms: Option<u64>,
    },
}

pub fn render_transcript(
    f: &mut Frame,
    area: Rect,
    items: &[TranscriptItem],
    scroll_offset: u16,
    auto_scroll: bool,
) {
    let mut lines = Vec::new();

    if items.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  ⚡ Agent Orchestra is ready.", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  Type a goal below and press ", Style::default().fg(Color::Gray)),
            Span::styled("[Enter]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" to initiate multi-agent collaboration.", Style::default().fg(Color::Gray)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  Hotkeys: ", Style::default().fg(Color::DarkGray)),
            Span::styled("[t] Change Topology   [m] Change Model   [c] Clear   [Tab] Switch Tab   [q] Quit", Style::default().fg(Color::DarkGray)),
        ]));
    }

    for item in items {
        lines.push(Line::from(""));
        match item {
            TranscriptItem::UserGoal { text, timestamp } => {
                lines.push(Line::from(vec![
                    Span::styled(" USER GOAL ", Style::default().bg(Color::Blue).fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  ({})", timestamp), Style::default().fg(Color::DarkGray)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled(format!("  {}", text), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]));
            }
            TranscriptItem::AgentOutput {
                agent_name,
                role,
                text,
                thoughts,
                is_streaming,
                ..
            } => {
                let badge_color = role.default_color();
                let spinner = if *is_streaming { " ⚡ STREAMING" } else { "" };

                lines.push(Line::from(vec![
                    Span::styled(format!(" {} {} ", role.icon(), agent_name.to_uppercase()), Style::default().bg(badge_color).fg(Color::Black).add_modifier(Modifier::BOLD)),
                    Span::styled(format!(" [{}]", role.name()), Style::default().fg(badge_color)),
                    Span::styled(spinner, Style::default().fg(Color::LightGreen).add_modifier(Modifier::SLOW_BLINK)),
                ]));

                if let Some(th) = thoughts {
                    if !th.is_empty() {
                        lines.push(Line::from(vec![
                            Span::styled("  💭 Reasoning Chain:", Style::default().fg(Color::Rgb(130, 130, 150)).add_modifier(Modifier::ITALIC)),
                        ]));
                        for th_line in th.lines() {
                            lines.push(Line::from(vec![
                                Span::styled(format!("    │ {}", th_line), Style::default().fg(Color::Rgb(110, 110, 130))),
                            ]));
                        }
                    }
                }

                // Render output lines
                for out_line in text.lines() {
                    if out_line.starts_with("```") {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {}", out_line), Style::default().fg(Color::Yellow)),
                        ]));
                    } else if out_line.starts_with('#') {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {}", out_line), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {}", out_line), Style::default().fg(Color::Rgb(220, 220, 230))),
                        ]));
                    }
                }
            }
            TranscriptItem::ToolExecution {
                agent_name,
                tool_name,
                args,
                output,
                is_error,
                duration_ms,
            } => {
                let (status_text, status_col) = if *is_error {
                    ("TOOL ERROR", Color::Red)
                } else {
                    ("TOOL SUCCESS", Color::LightGreen)
                };

                lines.push(Line::from(vec![
                    Span::styled(" 🛠️ TOOL CALL ", Style::default().bg(Color::Magenta).fg(Color::Black).add_modifier(Modifier::BOLD)),
                    Span::styled(format!(" {} invoked ", agent_name), Style::default().fg(Color::Gray)),
                    Span::styled(format!("`{}`", tool_name), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::styled(format!(" ({}ms) ", duration_ms), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("[{}]", status_text), Style::default().fg(status_col)),
                ]));

                let truncated_args = if args.len() > 100 { format!("{}...", &args[..100]) } else { args.clone() };
                lines.push(Line::from(vec![
                    Span::styled(format!("    Args: {}", truncated_args), Style::default().fg(Color::DarkGray)),
                ]));

                for out_line in output.lines().take(8) {
                    lines.push(Line::from(vec![
                        Span::styled(format!("    │ {}", out_line), Style::default().fg(Color::Rgb(160, 160, 180))),
                    ]));
                }
                if output.lines().count() > 8 {
                    lines.push(Line::from(vec![
                        Span::styled("    │ ... (remaining output truncated in view)", Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
            TranscriptItem::Milestone {
                step_title,
                duration_ms,
            } => {
                let dur_str = duration_ms.map(|d| format!(" [completed in {}ms]", d)).unwrap_or_default();
                lines.push(Line::from(vec![
                    Span::styled(" ──◆ ", Style::default().fg(Color::Cyan)),
                    Span::styled(step_title, Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                    Span::styled(dur_str, Style::default().fg(Color::DarkGray)),
                    Span::styled(" ───────────────────────", Style::default().fg(Color::Rgb(50, 50, 60))),
                ]));
            }
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(80, 85, 100)))
        .title(Span::styled(
            " Live Multi-Agent Workspace & Stream ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    let content_height = lines.len() as u16;
    let view_height = area.height.saturating_sub(2);
    let effective_scroll = if auto_scroll {
        content_height.saturating_sub(view_height)
    } else {
        scroll_offset.min(content_height.saturating_sub(view_height))
    };

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((effective_scroll, 0));

    f.render_widget(paragraph, area);
}
