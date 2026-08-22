use crate::tui::app::{ActiveTab, App, InputMode};
use crate::tui::layout::{grid_cells, grid_shape};
use crate::tui::widgets::{
    render_agent_pane, render_metrics_dashboard, render_transcript, PaneContext,
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
            Constraint::Length(3), // Command bar: models · query · status
            Constraint::Length(1), // Tab strip
            Constraint::Min(8),    // Workspace
        ])
        .split(size);

    // 1. Command bar — the query input, full width, with the models beside it.
    render_command_bar(f, root_chunks[0], app);
    render_tab_strip(f, root_chunks[1], app);

    // 2. Render Active Tab Content
    match app.active_tab {
        ActiveTab::Studio => render_agent_grid(f, root_chunks[2], app),
        ActiveTab::Telemetry => {
            render_metrics_dashboard(f, root_chunks[2], &app.metrics, &app.ordered_agents())
        }
        ActiveTab::AgentsConfig => render_agents_config_tab(f, root_chunks[2], app),
        ActiveTab::Blackboard => render_blackboard_and_logs_tab(f, root_chunks[2], app),
    }

    // 4. Render Modals if active
    match app.input_mode {
        InputMode::ModelSelectModal => render_model_modal(f, size, app),
        InputMode::TopologySelectModal => render_topology_modal(f, size, app),
        InputMode::HelpModal => render_help_modal(f, size),
        InputMode::PromptEditor => render_prompt_editor(f, size, app),
        _ => {}
    }
}

/// The command bar: model listing, the query input at full width, and run status.
///
/// The input moved from the bottom of the screen to the top, because it is the
/// one control the whole interface exists to serve.
fn render_command_bar(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(30), // models
            Constraint::Min(30),    // query
            Constraint::Length(30), // status
        ])
        .split(area);

    render_model_summary(f, chunks[0], app);
    render_query_input(f, chunks[1], app);
    render_run_status(f, chunks[2], app);
}

/// Which models are in play, collapsed to fit beside the input.
fn render_model_summary(f: &mut Frame, area: Rect, app: &App) {
    let mut distinct: Vec<&str> = app
        .ordered_agents()
        .iter()
        .map(|a| a.config.model.as_str())
        .collect();
    distinct.sort_unstable();
    distinct.dedup();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(58, 62, 76)))
        .title(Span::styled(
            format!(" Models ({}) ", distinct.len()),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));

    let text = match distinct.len() {
        0 => "none".to_string(),
        1 => distinct[0].to_string(),
        _ => distinct.join(" · "),
    };

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            text,
            Style::default().fg(Color::Rgb(170, 180, 205)),
        )))
        .block(block),
        area,
    );
}

/// The goal input, full width between the model list and the status.
fn render_query_input(f: &mut Frame, area: Rect, app: &App) {
    let editing = app.input_mode == InputMode::EditingPrompt;
    let border = if editing {
        Color::Yellow
    } else {
        Color::Rgb(58, 62, 76)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            " Goal ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    // Cursor position is measured in display columns: a CJK glyph is two wide,
    // so counting characters drifts the caret off the text it marks.
    let inner_width = area.width.saturating_sub(2) as usize;
    let prefix: String = app
        .prompt_input
        .chars()
        .take(app.input_cursor_pos)
        .collect();
    let cursor_col = UnicodeWidthStr::width(prefix.as_str());
    let h_scroll = cursor_col.saturating_sub(inner_width.saturating_sub(1));

    let content = if app.prompt_input.is_empty() && !editing {
        Span::styled(
            "Press [i] or [Enter] to type a goal…",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::styled(&app.prompt_input, Style::default().fg(Color::White))
    };

    f.render_widget(
        Paragraph::new(content)
            .block(block)
            .scroll((0, h_scroll as u16)),
        area,
    );

    if editing {
        f.set_cursor_position((area.x + 1 + (cursor_col - h_scroll) as u16, area.y + 1));
    }
}

/// Topology, progress and elapsed time.
fn render_run_status(f: &mut Frame, area: Rect, app: &App) {
    let (label, colour) = if app.is_running_workflow {
        let elapsed = app.metrics.global_elapsed_ms() as f64 / 1000.0;
        match app.step_progress {
            Some((current, total)) => (
                format!("STEP {current}/{total} · {elapsed:.0}s"),
                Color::LightGreen,
            ),
            None => (format!("RUNNING · {elapsed:.0}s"), Color::LightGreen),
        }
    } else {
        ("READY".to_string(), Color::Cyan)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(58, 62, 76)))
        .title(Span::styled(
            format!(" {} ", app.orchestrator.topology.name()),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            label,
            Style::default().fg(colour).add_modifier(Modifier::BOLD),
        )))
        .block(block)
        .alignment(Alignment::Center),
        area,
    );
}

/// One-line tab strip with the contextual key hints.
fn render_tab_strip(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(30), Constraint::Length(52)])
        .split(area);

    let tabs: Vec<Line> = ActiveTab::all()
        .iter()
        .map(|t| Line::from(t.title()))
        .collect();
    f.render_widget(
        Tabs::new(tabs)
            .select(app.active_tab as usize)
            .style(Style::default().fg(Color::Rgb(120, 122, 138)))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .divider("·"),
        chunks[0],
    );

    let hint = if app.is_running_workflow {
        "[Esc] cancel  [Tab] pane  [z] zoom"
    } else if app.zoomed {
        "[z]/[Esc] close  [Tab] pane"
    } else {
        "[Tab] pane  [z] zoom  [t] topo  [m] model  [?] help"
    };

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hint,
            Style::default().fg(Color::Rgb(120, 122, 138)),
        )))
        .alignment(Alignment::Right),
        chunks[1],
    );
}

/// The agent grid: every agent on screen at once, in an equal cell, with the
/// deliverable filling the spare slot.
fn render_agent_grid(f: &mut Frame, area: Rect, app: &App) {
    let agents = app.ordered_agents();
    if agents.is_empty() {
        return;
    }

    // One zoomed pane takes the whole area — a sixth of a terminal cannot show
    // a finished deliverable, and sometimes you need to read one agent closely.
    if app.zoomed {
        render_pane(f, area, app, app.focused_pane, &agents);
        return;
    }

    let panes = app.pane_count();
    let (cols, rows) = grid_shape(panes, area);
    let cells = grid_cells(area, cols, rows);

    for (index, cell) in cells.iter().enumerate().take(panes) {
        render_pane(f, *cell, app, index, &agents);
    }
}

/// Render pane `index`: an agent, or the deliverable in the final slot.
fn render_pane(
    f: &mut Frame,
    area: Rect,
    app: &App,
    index: usize,
    agents: &[&crate::core::agent::Agent],
) {
    match agents.get(index) {
        Some(agent) => {
            let view = app.view_for(&agent.config.id);
            render_agent_pane(
                f,
                area,
                &PaneContext {
                    agent,
                    view: &view,
                    metrics: app.metrics.agent_metrics.get(&agent.config.id),
                    focused: app.focused_pane == index,
                    spinner_idx: app.spinner_idx,
                    provider_online: app.provider_online,
                },
            );
        }
        None => render_deliverable_pane(f, area, app, app.focused_pane == index),
    }
}

/// The finished answer, and the files the run saved.
fn render_deliverable_pane(f: &mut Frame, area: Rect, app: &App, focused: bool) {
    let done = !app.deliverable.is_empty();
    let accent = if done {
        Color::LightGreen
    } else {
        Color::Rgb(58, 62, 76)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(if focused {
            ratatui::widgets::BorderType::Double
        } else {
            ratatui::widgets::BorderType::Plain
        })
        .border_style(Style::default().fg(if focused { Color::White } else { accent }))
        .title(Span::styled(
            " Deliverable ",
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    if app.files_written.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  workspace: {}", app.workspace.display()),
            Style::default().fg(Color::Rgb(120, 122, 138)),
        )));
    } else {
        for path in &app.files_written {
            lines.push(Line::from(vec![
                Span::styled("  ✓ ", Style::default().fg(Color::LightGreen)),
                Span::styled(path.clone(), Style::default().fg(Color::White)),
            ]));
        }
    }
    lines.push(Line::from(""));

    if done {
        for line in app.deliverable.lines() {
            lines.push(Line::from(Span::styled(
                line.to_string(),
                Style::default().fg(Color::Rgb(214, 216, 226)),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "  the finished answer appears here",
            Style::default()
                .fg(Color::Rgb(120, 122, 138))
                .add_modifier(Modifier::ITALIC),
        )));
    }

    // Show the head when zoomed and the tail while it streams in.
    let scroll = if app.zoomed {
        0
    } else {
        (lines.len() as u16).saturating_sub(inner.height)
    };

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        inner,
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

    // Right: the run's chronological activity. The per-agent panes show what
    // each agent is doing now; this is the only place the whole run reads in
    // order, which is what a log is for.
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
