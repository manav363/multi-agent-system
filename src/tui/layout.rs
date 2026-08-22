//! Geometry for the agent grid.
//!
//! Every agent is on screen at once, in a cell the same size as every other —
//! so the arrangement is derived from how many cells are needed rather than
//! hardcoded. The roster is configurable, so "five agents" is not a safe
//! assumption; four, five or seven all have to tile sensibly.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Narrowest a cell may be before the grid drops a column.
const MIN_CELL_WIDTH: u16 = 34;
/// Shortest a cell may be before the grid drops a row.
const MIN_CELL_HEIGHT: u16 = 7;

/// Rows and columns for `cells` panes inside `area`.
///
/// Aims for the squarest arrangement that fits, then backs off a column at a
/// time while cells would be too narrow to read. Every cell is the same size;
/// any leftover slots are spares the caller can fill.
pub fn grid_shape(cells: usize, area: Rect) -> (usize, usize) {
    if cells == 0 {
        return (0, 0);
    }

    let max_cols = ((area.width / MIN_CELL_WIDTH).max(1) as usize).min(cells);
    let max_rows = ((area.height / MIN_CELL_HEIGHT).max(1) as usize).min(cells);

    // Squarest shape first: a 3x2 beats a 6x1 for six cells.
    let ideal_cols = (cells as f64).sqrt().ceil() as usize;
    let cols = ideal_cols.clamp(1, max_cols);
    let rows = cells.div_ceil(cols);

    // Too tall to fit: widen instead, accepting narrower cells.
    if rows > max_rows {
        let widened = cells.div_ceil(max_rows).min(cells);
        return (widened, cells.div_ceil(widened));
    }

    (cols, rows)
}

/// Split `area` into equal cells, row by row, left to right.
///
/// Returns exactly `rows * cols` rectangles — the caller decides which are
/// agents and which are spare.
pub fn grid_cells(area: Rect, cols: usize, rows: usize) -> Vec<Rect> {
    if cols == 0 || rows == 0 {
        return Vec::new();
    }

    let row_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Ratio(1, rows as u32); rows])
        .split(area);

    row_areas
        .iter()
        .flat_map(|row| {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints(vec![Constraint::Ratio(1, cols as u32); cols])
                .split(*row)
                .to_vec()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(w: u16, h: u16) -> Rect {
        Rect::new(0, 0, w, h)
    }

    #[test]
    fn five_agents_plus_a_deliverable_tile_as_three_by_two() {
        // The layout this design is built around: six equal cells.
        assert_eq!(grid_shape(6, area(200, 50)), (3, 2));
    }

    #[test]
    fn a_four_agent_roster_tiles_as_two_by_two() {
        assert_eq!(grid_shape(4, area(200, 50)), (2, 2));
    }

    #[test]
    fn the_shape_is_as_square_as_it_can_be() {
        for (cells, expected) in [(1, (1, 1)), (2, (2, 1)), (3, (2, 2)), (9, (3, 3))] {
            assert_eq!(grid_shape(cells, area(400, 100)), expected, "cells={cells}");
        }
    }

    #[test]
    fn every_cell_is_accounted_for() {
        for cells in 1..=12 {
            let (cols, rows) = grid_shape(cells, area(400, 100));
            assert!(cols * rows >= cells, "cells={cells} lost a slot");
            // And never wastes a whole row of spares.
            assert!(cols * rows < cells + cols, "cells={cells} over-allocated");
        }
    }

    #[test]
    fn a_narrow_terminal_drops_columns_rather_than_squeezing() {
        // 80 columns cannot hold three readable cells.
        let (cols, _) = grid_shape(6, area(80, 50));
        assert!(cols <= 2, "got {cols} columns in 80 chars");
        // One very narrow terminal still renders a single column.
        assert_eq!(grid_shape(6, area(30, 50)).0, 1);
    }

    #[test]
    fn a_short_terminal_widens_rather_than_stacking_off_screen() {
        // Room for two rows only, so six cells need three columns.
        let (cols, rows) = grid_shape(6, area(200, 16));
        assert!(rows <= 2, "got {rows} rows in 16 lines");
        assert!(cols * rows >= 6);
    }

    #[test]
    fn cells_are_equal_and_cover_the_area() {
        let full = area(180, 48);
        let cells = grid_cells(full, 3, 2);
        assert_eq!(cells.len(), 6);

        let widths: Vec<u16> = cells.iter().map(|c| c.width).collect();
        let heights: Vec<u16> = cells.iter().map(|c| c.height).collect();
        // Ratio splitting can differ by a single column on rounding.
        assert!(widths.iter().max().unwrap() - widths.iter().min().unwrap() <= 1);
        assert!(heights.iter().max().unwrap() - heights.iter().min().unwrap() <= 1);

        assert_eq!(cells[0].y, cells[2].y, "first row shares a top edge");
        assert!(cells[3].y > cells[0].y, "second row sits below the first");
    }

    #[test]
    fn cells_do_not_overlap() {
        let cells = grid_cells(area(180, 48), 3, 2);
        for (i, a) in cells.iter().enumerate() {
            for b in cells.iter().skip(i + 1) {
                let disjoint = a.x + a.width <= b.x
                    || b.x + b.width <= a.x
                    || a.y + a.height <= b.y
                    || b.y + b.height <= a.y;
                assert!(disjoint, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn an_empty_grid_is_handled_without_panicking() {
        assert_eq!(grid_shape(0, area(100, 40)), (0, 0));
        assert!(grid_cells(area(100, 40), 0, 0).is_empty());
    }
}
