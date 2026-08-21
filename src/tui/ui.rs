use crate::tui::app::{ActiveTab, App, InputMode};
use crate::tui::widgets::{
    render_agent_pipeline_cards, render_metrics_dashboard, render_transcript,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

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
        InputMode::PromptEditor => render_prompt_editor(f, size, app),
        _ => {}
    }
}

fn render_header_and_tabs(f: &mut Frame, area: Rect, app: &App) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(24),
            Constraint::Min(30),
            Constraint::Length(52),
        ])
        .split(area);

    // Left: Brand Logo
    let title_line = Line::from(vec![
        Span::styled(
            "⚡ AGENT ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "ORCHESTRA",
            Style::default()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" v0.1", Style::default().fg(Color::DarkGray)),
    ]);
    let title_block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::Rgb(60, 65, 80)));
    f.render_widget(
        Paragraph::new(title_line).block(title_block),
        header_chunks[0],
    );

    // Middle: Tab Bar
    let tab_titles: Vec<Line> = ActiveTab::all()
        .iter()
        .map(|t| Line::from(t.title()))
        .collect();

    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::Rgb(60, 65, 80))),
        )
        .select(app.active_tab as usize)
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, header_chunks[1]);

    // Right: Active Topology & Live Status
    let active_model = app
        .available_models
        .get(app.selected_model_idx)
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let (status_str, status_color) = if app.is_running_workflow {
        let elapsed = app.metrics.global_elapsed_ms() as f64 / 1000.0;
        match app.step_progress {
            Some((current, total)) => (
                format!("⚡ STEP {}/{} · {:.0}s", current, total, elapsed),
                Color::Green,
            ),
            None => (format!("⚡ RUNNING · {:.0}s", elapsed), Color::Green),
        }
    } else {
        ("● READY".to_string(), Color::Cyan)
    };

    let right_info = Line::from(vec![
        Span::styled(
            format!("{} ", app.orchestrator.topology.name()),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("│ {} │ ", active_model),
            Style::default().fg(Color::Rgb(170, 170, 190)),
        ),
        Span::styled(
            status_str,
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
    ]);

    let right_block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::Rgb(60, 65, 80)));

    f.render_widget(
        Paragraph::new(right_info)
            .block(right_block)
            .alignment(Alignment::Right),
        header_chunks[2],
    );
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

    // The widget reports back what it measured; stash it so key handling can
    // clamp scrolling to content that actually exists.
    let viewport = render_transcript(
        f,
        chunks[1],
        &app.transcript_items,
        app.scroll_offset,
        app.auto_scroll,
        app.spinner_idx,
    );
    app.transcript_viewport.set(viewport);
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
                Span::styled(
                    &agent.config.name,
                    Style::default()
                        .fg(agent.config.role.default_color())
                        .add_modifier(if is_sel {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(
            " Agent Roster ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    let list = List::new(items).block(list_block);
    f.render_widget(list, chunks[0]);

    // Right: Selected Agent Detail & Prompt
    let sel_agent = agents.get(app.selected_agent_idx).unwrap_or(&agents[0]);

    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(70, 75, 90)))
        .title(Span::styled(
            format!(" Configuration: {} ", sel_agent.config.name),
            Style::default()
                .fg(sel_agent.config.role.default_color())
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            " [e] edit prompt · [m] change model · [←/→] select agent ",
            Style::default().fg(Color::DarkGray),
        ));

    let tools_str = if sel_agent.config.enabled_tools.is_empty() {
        "None (Pure Inference)".to_string()
    } else {
        sel_agent.config.enabled_tools.join(", ")
    };

    let details = vec![
        Line::from(vec![
            Span::styled("Role:         ", Style::default().fg(Color::Gray)),
            Span::styled(
                sel_agent.config.role.name(),
                Style::default()
                    .fg(sel_agent.config.role.default_color())
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Model:        ", Style::default().fg(Color::Gray)),
            Span::styled(&sel_agent.config.model, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Temperature:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2}", sel_agent.config.temperature),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  |  Max Tokens: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:?}", sel_agent.config.max_tokens),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Tools:        ", Style::default().fg(Color::Gray)),
            Span::styled(tools_str, Style::default().fg(Color::LightMagenta)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "── System Prompt Instructions ────────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        )]),
        Line::from(""),
    ];

    let mut all_lines = details;
    for l in sel_agent.config.system_prompt.lines() {
        all_lines.push(Line::from(vec![Span::styled(
            format!("  {}", l),
            Style::default().fg(Color::Rgb(210, 210, 220)),
        )]));
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
        .title(Span::styled(
            " Shared Memory & Blackboard (Live) ",
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            format!(" files written to {} ", app.workspace.display()),
            Style::default().fg(Color::DarkGray),
        ));

    let mut blackboard_lines = vec![
        Line::from(vec![Span::styled(
            "  Inter-agent shared context memory:",
            Style::default().fg(Color::DarkGray),
        )]),
        Line::from(""),
    ];

    if app.blackboard_snapshot.is_empty() {
        blackboard_lines.push(Line::from(vec![
            Span::styled("  (empty) ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Run a workflow to populate blackboard.",
                Style::default().fg(Color::Rgb(120, 120, 130)),
            ),
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
                    Span::styled(
                        format!("{}: ", key),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("({} bytes)", value.len()),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                blackboard_lines.push(Line::from(vec![Span::styled(
                    format!("    {}{}", preview, suffix),
                    Style::default().fg(Color::Rgb(180, 180, 190)),
                )]));
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
        .title(Span::styled(
            " System Event Logs ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

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
            ListItem::new(Line::from(vec![Span::styled(
                format!("  {}", log),
                Style::default().fg(color),
            )]))
        })
        .collect();

    let logs_list = List::new(log_items).block(logs_block);
    f.render_widget(logs_list, chunks[1]);
}

fn render_input_bar(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(12),
            Constraint::Min(20),
            Constraint::Length(45),
        ])
        .split(area);

    let (mode_badge, mode_color) = match app.input_mode {
        InputMode::EditingPrompt => (" [INPUT] ", Color::Yellow),
        InputMode::PromptEditor => (" [EDIT] ", Color::Yellow),
        _ => (" [NORMAL] ", Color::Cyan),
    };

    let mode_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(mode_color));

    f.render_widget(
        Paragraph::new(Span::styled(
            mode_badge,
            Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
        ))
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
        .title(Span::styled(
            " Prompt / Goal Input ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    // Cursor position is measured in display columns, not characters: a CJK
    // glyph is two columns wide and an emoji can be more, so counting
    // characters drifts the caret away from the text it is meant to sit on.
    let inner_width = chunks[1].width.saturating_sub(2) as usize;
    let prefix: String = app
        .prompt_input
        .chars()
        .take(app.input_cursor_pos)
        .collect();
    let cursor_col = UnicodeWidthStr::width(prefix.as_str());
    // Keep the caret in view on input longer than the box.
    let h_scroll = cursor_col.saturating_sub(inner_width.saturating_sub(1));

    let prompt_display =
        if app.prompt_input.is_empty() && app.input_mode != InputMode::EditingPrompt {
            Span::styled(
                "Press [i] or [Enter] to type a goal…",
                Style::default().fg(Color::DarkGray),
            )
        } else {
            Span::styled(&app.prompt_input, Style::default().fg(Color::White))
        };

    f.render_widget(
        Paragraph::new(prompt_display)
            .block(input_block)
            .scroll((0, h_scroll as u16)),
        chunks[1],
    );

    if app.input_mode == InputMode::EditingPrompt {
        f.set_cursor_position((
            chunks[1].x + 1 + (cursor_col - h_scroll) as u16,
            chunks[1].y + 1,
        ));
    }

    // Right shortcut hints
    let shortcuts = if app.is_running_workflow {
        Line::from(vec![
            Span::styled(
                "[Esc]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Cancel run  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[j/k]",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Scroll  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[G]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Follow", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                "[t]",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Topo  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[m]",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Model  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[c]",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Clear  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[?]",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Help  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "[q]",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
        ])
    };

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
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
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
        .title(Span::styled(
            " Select Active Open-Source Model (↑/↓ Enter) ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));

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
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(marker, Style::default().fg(Color::Yellow)),
                    Span::styled(topo.name(), style),
                ]),
                Line::from(vec![Span::styled(
                    format!("    {}", topo.description()),
                    Style::default().fg(Color::DarkGray),
                )]),
            ])
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta))
        .title(Span::styled(
            " Select Swarm Topology (↑/↓ Enter) ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(List::new(items).block(block), popup_area);
}

/// Full-height editor for the selected agent's system prompt.
fn render_prompt_editor(f: &mut Frame, area: Rect, app: &App) {
    let popup = centered_rect(78, 82, area);
    f.render_widget(Clear, popup);

    let agents = app.ordered_agents();
    let name = agents
        .get(app.selected_agent_idx)
        .map(|a| a.config.name.as_str())
        .unwrap_or("agent");

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            format!(" Editing system prompt · {name} "),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            format!(
                " Ctrl+S save to {} · Esc cancel ",
                app.roster_path.display()
            ),
            Style::default().fg(Color::DarkGray),
        ));

    let inner_height = popup.height.saturating_sub(2) as usize;
    let lines: Vec<&str> = app.prompt_editor.split('\n').collect();

    // Cursor's line and column, so the view can follow it.
    let mut remaining = app.prompt_editor_cursor;
    let (mut cursor_line, mut cursor_col) = (0usize, 0usize);
    for (idx, line) in lines.iter().enumerate() {
        let len = line.chars().count();
        if remaining <= len {
            cursor_line = idx;
            cursor_col = remaining;
            break;
        }
        remaining -= len + 1; // the newline itself
        cursor_line = idx + 1;
    }

    let scroll = cursor_line.saturating_sub(inner_height.saturating_sub(1));
    let body: Vec<Line> = lines
        .iter()
        .map(|l| {
            Line::from(Span::styled(
                l.to_string(),
                Style::default().fg(Color::White),
            ))
        })
        .collect();

    f.render_widget(
        Paragraph::new(body).block(block).scroll((scroll as u16, 0)),
        popup,
    );

    f.set_cursor_position((
        popup.x + 1 + cursor_col as u16,
        popup.y + 1 + (cursor_line - scroll) as u16,
    ));
}

fn render_help_modal(f: &mut Frame, area: Rect) {
    let popup_area = centered_rect(64, 72, area);
    f.render_widget(Clear, popup_area);

    fn row(keys: &'static str, desc: &'static str) -> Line<'static> {
        Line::from(vec![
            Span::styled(
                format!("  {:<16}", keys),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(desc, Style::default().fg(Color::Gray)),
        ])
    }

    fn section(title: &'static str) -> Line<'static> {
        Line::from(vec![Span::styled(
            title,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )])
    }

    let lines = vec![
        Line::from(vec![Span::styled(
            "⚡ Agent Orchestra — terminal multi-agent runner",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        section("Running a goal"),
        row("i / Enter", "Focus the prompt input"),
        row("Enter", "Submit the goal and start the pipeline"),
        row(
            "Esc",
            "Unfocus input, close a modal, or cancel a running workflow",
        ),
        Line::from(""),
        section("Navigation"),
        row(
            "Tab / 1-4",
            "Studio · Telemetry · Agent Roster · Blackboard",
        ),
        row("j / k / ↑ ↓", "Scroll the transcript"),
        row("PgUp / PgDn", "Scroll by a full page"),
        row("g / G", "Jump to top / bottom (G re-enables follow mode)"),
        row("Mouse wheel", "Scroll the transcript"),
        row("← / →", "Select an agent card"),
        Line::from(""),
        section("Configuration"),
        row("t", "Choose the swarm topology"),
        row("m", "Change the model (per-agent on the Roster tab)"),
        row("c", "Clear the transcript"),
        row("s", "Export this run as Markdown"),
        row("e", "Edit the selected agent's prompt (Roster tab)"),
        Line::from(""),
        section("Editing text"),
        row("Ctrl+S", "Save an edited agent prompt to the roster file"),
        row("Ctrl+W", "Delete the previous word"),
        row("Ctrl+U", "Delete to the start of the line"),
        row("Home / End", "Jump to start / end of the line"),
        Line::from(""),
        section("Exit"),
        row("q / Ctrl+C", "Quit"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Press Esc or Enter to close.",
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::LightGreen))
        .title(Span::styled(
            " Help ",
            Style::default()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        popup_area,
    );
}
