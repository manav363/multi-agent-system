use crate::tui::app::{ActiveTab, App, InputMode};
use crate::tui::widgets::{render_agent_pipeline_cards, render_metrics_dashboard, render_transcript};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::Frame;

pub fn render_app_ui(f: &mut Frame, app: &App) {
    let size = f.area();

    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header & Tabs
            Constraint::Min(10),   // Active Workspace Tab Content
            Constraint::Length(3), // Input Prompt Box & Status Bar
        ])
        .split(size);

    // 1. Render Header & Tab Bar
    render_header_and_tabs(f, root_chunks[0], app);

    // 2. Render Active Tab Content
    match app.active_tab {
        ActiveTab::Studio => render_studio_tab(f, root_chunks[1], app),
        ActiveTab::Telemetry => {
            render_metrics_dashboard(f, root_chunks[1], &app.metrics, &app.ordered_agents())
        }
        ActiveTab::AgentsConfig => render_agents_config_tab(f, root_chunks[1], app),
        ActiveTab::Blackboard => render_blackboard_and_logs_tab(f, root_chunks[1], app),
    }

    // 3. Render Bottom Input & Status Bar
    render_input_bar(f, root_chunks[2], app);

    // 4. Render Modals if active
    match app.input_mode {
        InputMode::ModelSelectModal => render_model_modal(f, size, app),
        InputMode::TopologySelectModal => render_topology_modal(f, size, app),
        InputMode::HelpModal => render_help_modal(f, size),
        _ => {}
    }
}

fn render_header_and_tabs(f: &mut Frame, area: Rect, app: &App) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(40), Constraint::Length(36)])
        .split(area);

    // Left: Brand Logo
    let title_line = Line::from(vec![
        Span::styled("⚡ AGENT ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("ORCHESTRA", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
        Span::styled(" v0.1", Style::default().fg(Color::DarkGray)),
    ]);
    let title_block = Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(Color::Rgb(60, 65, 80)));
    f.render_widget(Paragraph::new(title_line).block(title_block), header_chunks[0]);

    // Middle: Tab Bar
    let tab_titles: Vec<Line> = ActiveTab::all()
        .iter()
        .map(|t| Line::from(t.title()))
        .collect();

    let tabs = Tabs::new(tab_titles)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(Color::Rgb(60, 65, 80))))
        .select(app.active_tab as usize)
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    f.render_widget(tabs, header_chunks[1]);

    // Right: Active Topology & Live Status
    let active_model = app.available_models.get(app.selected_model_idx).cloned().unwrap_or_else(|| "default".to_string());
    let status_str = if app.is_running_workflow { "⚡ RUNNING" } else { "● READY" };
    let status_color = if app.is_running_workflow { Color::Green } else { Color::Cyan };

    let right_info = Line::from(vec![
        Span::styled(format!("{:<10} ", app.orchestrator.topology.name()), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::styled(format!("| {} | ", active_model), Style::default().fg(Color::Rgb(170, 170, 190))),
        Span::styled(status_str, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
    ]);

    let right_block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::Rgb(60, 65, 80)));

    f.render_widget(Paragraph::new(right_info).block(right_block).alignment(Alignment::Right), header_chunks[2]);
}

fn render_studio_tab(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // Agent Cards Pipeline
            Constraint::Min(8),    // Live Transcript
        ])
        .split(area);

    let agents = app.ordered_agents();
    render_agent_pipeline_cards(
        f,
        chunks[0],
        &agents,
        &app.metrics.agent_metrics,
        app.spinner_idx,
        Some(app.selected_agent_idx),
    );

    render_transcript(
        f,
        chunks[1],
        &app.transcript_items,
        app.scroll_offset,
        app.auto_scroll,
    );
}

fn render_agents_config_tab(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(40)])
        .split(area);

    let agents = app.ordered_agents();

    // Left: Agent Roster List
    let items: Vec<ListItem> = agents
        .iter()
        .enumerate()
        .map(|(idx, agent)| {
            let is_sel = idx == app.selected_agent_idx;
            let marker = if is_sel { "▶ " } else { "  " };
            let line = Line::from(vec![
                Span::styled(marker, Style::default().fg(Color::Yellow)),
                Span::styled(format!("{} ", agent.config.role.icon()), Style::default()),
                Span::styled(&agent.config.name, Style::default().fg(agent.config.role.default_color()).add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() })),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Agent Roster ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    let list = List::new(items).block(list_block);
    f.render_widget(list, chunks[0]);

    // Right: Selected Agent Detail & Prompt
    let sel_agent = agents.get(app.selected_agent_idx).unwrap_or(&agents[0]);

    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(
            format!(" Configuration: {} ", sel_agent.config.name),
            Style::default().fg(sel_agent.config.role.default_color()).add_modifier(Modifier::BOLD),
        ));

    let tools_str = if sel_agent.config.enabled_tools.is_empty() {
        "None (Pure Inference)".to_string()
    } else {
        sel_agent.config.enabled_tools.join(", ")
    };

    let details = vec![
        Line::from(vec![
            Span::styled("Role:         ", Style::default().fg(Color::Gray)),
            Span::styled(sel_agent.config.role.name(), Style::default().fg(sel_agent.config.role.default_color()).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Model:        ", Style::default().fg(Color::Gray)),
            Span::styled(&sel_agent.config.model, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Temperature:  ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.2}", sel_agent.config.temperature), Style::default().fg(Color::Yellow)),
            Span::styled("  |  Max Tokens: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:?}", sel_agent.config.max_tokens), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Tools:        ", Style::default().fg(Color::Gray)),
            Span::styled(tools_str, Style::default().fg(Color::LightMagenta)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("── System Prompt Instructions ────────────────────────────────────", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
    ];

    let mut all_lines = details;
    for l in sel_agent.config.system_prompt.lines() {
        all_lines.push(Line::from(vec![
            Span::styled(format!("  {}", l), Style::default().fg(Color::Rgb(210, 210, 220))),
        ]));
    }

    let detail_paragraph = Paragraph::new(all_lines)
        .block(right_block)
        .wrap(Wrap { trim: false });

    f.render_widget(detail_paragraph, chunks[1]);
}

fn render_blackboard_and_logs_tab(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // Left: Blackboard artifacts (live data)
    let blackboard_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" Shared Memory & Blackboard (Live) ", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)));

    let mut blackboard_lines = vec![
        Line::from(vec![
            Span::styled("  Inter-agent shared context memory:", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
    ];

    if app.blackboard_snapshot.is_empty() {
        blackboard_lines.push(Line::from(vec![
            Span::styled("  (empty) ", Style::default().fg(Color::DarkGray)),
            Span::styled("Run a workflow to populate blackboard.", Style::default().fg(Color::Rgb(120, 120, 130))),
        ]));
    } else {
        // Sort keys for consistent display
        let mut keys: Vec<&String> = app.blackboard_snapshot.keys().collect();
        keys.sort();

        for key in keys {
            if let Some(value) = app.blackboard_snapshot.get(key) {
                let preview: String = value
                    .chars()
                    .take(80)
                    .map(|c| if c == '\n' { ' ' } else { c })
                    .collect();
                let suffix = if value.len() > 80 { "..." } else { "" };

                blackboard_lines.push(Line::from(vec![
                    Span::styled("  ● ", Style::default().fg(Color::Cyan)),
                    Span::styled(format!("{}: ", key), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::styled(
                        format!("({} bytes)", value.len()),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                blackboard_lines.push(Line::from(vec![
                    Span::styled(format!("    {}{}", preview, suffix), Style::default().fg(Color::Rgb(180, 180, 190))),
                ]));
                blackboard_lines.push(Line::from(""));
            }
        }
    }

    f.render_widget(
        Paragraph::new(blackboard_lines)
            .block(blackboard_block)
            .wrap(Wrap { trim: false }),
        chunks[0],
    );

    // Right: System logs
    let logs_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(" System Event Logs ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    let log_items: Vec<ListItem> = app
        .system_logs
        .iter()
        .map(|log| {
            let color = if log.contains("ERROR") {
                Color::Red
            } else if log.contains("WARN") {
                Color::Yellow
            } else {
                Color::Rgb(160, 160, 170)
            };
            ListItem::new(Line::from(vec![Span::styled(format!("  {}", log), Style::default().fg(color))]))
        })
        .collect();

    let logs_list = List::new(log_items).block(logs_block);
    f.render_widget(logs_list, chunks[1]);
}

fn render_input_bar(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(12), Constraint::Min(20), Constraint::Length(45)])
        .split(area);

    let (mode_badge, mode_color) = match app.input_mode {
        InputMode::EditingPrompt => (" [INPUT] ", Color::Yellow),
        _ => (" [NORMAL] ", Color::Cyan),
    };

    let mode_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(mode_color));

    f.render_widget(
        Paragraph::new(Span::styled(mode_badge, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)))
            .block(mode_block)
            .alignment(Alignment::Center),
        chunks[0],
    );

    // Prompt input text box
    let input_border_color = if app.input_mode == InputMode::EditingPrompt {
        Color::Yellow
    } else {
        Color::Rgb(70, 75, 90)
    };

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(input_border_color))
        .title(Span::styled(" Prompt / Goal Input ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    let prompt_display = if app.prompt_input.is_empty() && app.input_mode != InputMode::EditingPrompt {
        Span::styled("Press [i] or [Enter] to type goal...", Style::default().fg(Color::DarkGray))
    } else {
        Span::styled(&app.prompt_input, Style::default().fg(Color::White))
    };

    let prompt_p = Paragraph::new(prompt_display).block(input_block);
    f.render_widget(prompt_p, chunks[1]);

    if app.input_mode == InputMode::EditingPrompt {
        let cursor_x = chunks[1].x + 1 + app.input_cursor_pos as u16;
        let cursor_y = chunks[1].y + 1;
        f.set_cursor_position((cursor_x, cursor_y));
    }

    // Right shortcut hints
    let shortcuts = Line::from(vec![
        Span::styled("[t]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::styled(" Topo  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[m]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Model  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[c]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Clear  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[?]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled(" Help  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[q]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
    ]);

    let shortcuts_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)));

    f.render_widget(
        Paragraph::new(shortcuts)
            .block(shortcuts_block)
            .alignment(Alignment::Right),
        chunks[2],
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn render_model_modal(f: &mut Frame, area: Rect, app: &App) {
    let popup_area = centered_rect(50, 45, area);
    f.render_widget(Clear, popup_area);

    let items: Vec<ListItem> = app
        .available_models
        .iter()
        .enumerate()
        .map(|(idx, model)| {
            let is_sel = idx == app.selected_model_idx;
            let marker = if is_sel { "▶ " } else { "  " };
            let style = if is_sel {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(vec![
                Span::styled(marker, Style::default().fg(Color::Yellow)),
                Span::styled(model, style),
            ]))
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(" Select Active Open-Source Model (↑/↓ Enter) ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    f.render_widget(List::new(items).block(block), popup_area);
}

fn render_topology_modal(f: &mut Frame, area: Rect, app: &App) {
    let popup_area = centered_rect(65, 55, area);
    f.render_widget(Clear, popup_area);

    let topologies = App::topologies();
    let items: Vec<ListItem> = topologies
        .iter()
        .enumerate()
        .map(|(idx, topo)| {
            let is_sel = idx == app.selected_topology_idx;
            let marker = if is_sel { "▶ " } else { "  " };
            let style = if is_sel {
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(marker, Style::default().fg(Color::Yellow)),
                    Span::styled(topo.name(), style),
                ]),
                Line::from(vec![
                    Span::styled(format!("    {}", topo.description()), Style::default().fg(Color::DarkGray)),
                ]),
            ])
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta))
        .title(Span::styled(" Select Swarm Topology (↑/↓ Enter) ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)));

    f.render_widget(List::new(items).block(block), popup_area);
}

fn render_help_modal(f: &mut Frame, area: Rect) {
    let popup_area = centered_rect(60, 60, area);
    f.render_widget(Clear, popup_area);

    let lines = vec![
        Line::from(vec![
            Span::styled("⚡ Agent Orchestra - Terminal Multi-Agent System", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Navigation & Hotkeys:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  [i] / [Enter]  ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Focus prompt input box", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [Esc]          ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Unfocus input / Close modals", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [Tab] / [1-4]  ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Switch between Studio, Telemetry, Prompts, Logs", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [t]            ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Select Swarm Topology (Hierarchical, Pipeline, Debate, Direct)", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [m]            ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Switch active Local Model (Ollama / llama.cpp / vLLM)", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [c]            ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Clear workspace transcript", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [j] / [k]      ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Scroll transcript up / down", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled("  [q] / [Ctrl+C] ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Exit application", Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Press [Esc] or [Enter] to close this help window.", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::LightGreen))
        .title(Span::styled(" Help & Architecture Guide ", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)));

    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), popup_area);
}
