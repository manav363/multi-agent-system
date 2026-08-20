use crate::core::agent::Agent;
use crate::metrics::tracker::MetricsTracker;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Sparkline, Table, Wrap};
use ratatui::Frame;

pub fn render_metrics_dashboard(
    f: &mut Frame,
    area: Rect,
    tracker: &MetricsTracker,
    agents: &[&Agent],
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // Summary & Sparkline
            Constraint::Length(10), // Per-agent latency table
            Constraint::Min(8),     // Waterfall Timeline
        ])
        .split(area);

    // 1. Top Section: Global throughput & Sparkline
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(38), Constraint::Min(20)])
        .split(chunks[0]);

    let elapsed_ms = tracker.global_elapsed_ms();
    let total_tokens = tracker.total_workflow_tokens;
    let avg_tps = tracker.overall_average_tps();
    let peak_tps = tracker.tps_history.iter().copied().max().unwrap_or(0);

    let summary_lines = vec![
        Line::from(vec![
            Span::styled("Total Elapsed:   ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.2}s", elapsed_ms as f64 / 1000.0), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Tokens Streamed: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", total_tokens), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Average Speed:   ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.1} tok/s", avg_tps), Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Peak Speed:      ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{} tok/s", peak_tps), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
    ];

    let summary_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Cluster Telemetry ", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)));

    f.render_widget(Paragraph::new(summary_lines).block(summary_block), top_chunks[0]);

    let sparkline_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Real-time Token Throughput (tok/s) ", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)));

    let sparkline = Sparkline::default()
        .block(sparkline_block)
        .data(&tracker.tps_history)
        .style(Style::default().fg(Color::LightGreen));

    f.render_widget(sparkline, top_chunks[1]);

    // 2. Middle Section: Per-agent metrics table
    let table_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Per-Agent Latency & Performance Breakdown ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    let header_cells = ["Agent Role", "Assigned Model", "TTFT (Latency)", "Tokens", "Avg TPS", "Tool Exec (ms)"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows: Vec<Row> = agents
        .iter()
        .map(|agent| {
            let m = tracker.agent_metrics.get(&agent.config.id);
            let ttft_str = m.and_then(|x| x.ttft_ms).map(|ms| format!("{} ms", ms)).unwrap_or_else(|| "-".to_string());
            let tokens = m.map(|x| x.total_tokens).unwrap_or(0);
            let tps = m.map(|x| x.avg_tps).unwrap_or(0.0);
            let tool_dur = m.map(|x| x.total_tool_duration_ms).unwrap_or(0);

            let cells = vec![
                Cell::from(format!("{} {}", agent.config.role.icon(), agent.config.name)).style(Style::default().fg(agent.config.role.default_color())),
                Cell::from(agent.config.model.clone()).style(Style::default().fg(Color::Gray)),
                Cell::from(ttft_str).style(Style::default().fg(Color::Cyan)),
                Cell::from(format!("{}", tokens)).style(Style::default().fg(Color::White)),
                Cell::from(format!("{:.1} tok/s", tps)).style(Style::default().fg(if tps > 0.0 { Color::LightGreen } else { Color::DarkGray })),
                Cell::from(format!("{} ms", tool_dur)).style(Style::default().fg(if tool_dur > 0 { Color::LightMagenta } else { Color::DarkGray })),
            ];
            Row::new(cells).height(1)
        })
        .collect();

    let widths = [
        Constraint::Percentage(25),
        Constraint::Percentage(20),
        Constraint::Percentage(15),
        Constraint::Percentage(12),
        Constraint::Percentage(13),
        Constraint::Percentage(15),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(table_block);

    f.render_widget(table, chunks[1]);

    // 3. Bottom Section: Step Waterfall Timeline
    let waterfall_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Step Execution Waterfall & Gantt Timeline ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    let mut waterfall_lines = Vec::new();
    if tracker.waterfall_spans.is_empty() {
        waterfall_lines.push(Line::from(vec![
            Span::styled("  No completed steps yet. Run a prompt to visualize the multi-agent waterfall timeline.", Style::default().fg(Color::DarkGray)),
        ]));
    } else {
        for span in &tracker.waterfall_spans {
            let bar_len = ((span.duration_ms as f64 / elapsed_ms.max(1) as f64) * 30.0).clamp(1.0, 35.0) as usize;
            let bar = "█".repeat(bar_len);

            waterfall_lines.push(Line::from(vec![
                Span::styled(format!(" Step {}: ", span.step_index), Style::default().fg(Color::Yellow)),
                Span::styled(format!("{:<30} ", span.title), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::styled(format!("[{}] ", span.agent_name), Style::default().fg(Color::Cyan)),
                Span::styled(format!("{:>5}ms ", span.duration_ms), Style::default().fg(Color::LightCyan)),
                Span::styled(bar, Style::default().fg(Color::LightGreen)),
                Span::styled(format!(" ({} tok, {:.1} tps)", span.tokens_generated, span.avg_tps), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(waterfall_lines)
            .block(waterfall_block)
            .wrap(Wrap { trim: false }),
        chunks[2],
    );
}
