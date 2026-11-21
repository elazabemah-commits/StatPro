# StatPro — Detailed User Manual
**Complete guide to every tab, menu, button, and major function**

File: `StatPro_coherent.py`  
Edition: coherent single-file

---

## 1. Purpose of the program

StatPro is a desktop application for agricultural and biological experiments. It:

- Checks residual assumptions (normality and homogeneity of variance)
- Selects a scale (raw or transformed) using a fixed ladder
- Runs the appropriate test (ANOVA, Welch, HC3, Kruskal-Wallis, or aligned-rank transform)
- Produces post-hoc tests and compact letter displays
- Handles multi-factor and simple random-intercept models
- Exports tables and a ready-to-paste methods paragraph

The Assumptions tab and the ANOVA/KW tab always use the **same one-way residual checks** on the separation factor, so their conclusions cannot contradict each other.

---

## 2. Starting the program

```bash
python StatPro_coherent.py
```

Built-in tests:
```bash
python StatPro_coherent.py --selftest
```

If packages are missing the program prints exactly which ones to install.

---

## 3. Main window layout

The program uses a notebook with five tabs:

1. Data  
2. Assumptions  
3. ANOVA / KW  
4. Two-Sample Tests  
5. Figures  

A status bar at the bottom shows progress and messages.

---

## 4. Data tab — detailed

### Buttons and actions
- **Load Excel**: Opens a file dialog. Reads the first sheet into a pandas DataFrame and displays it in an editable Treeview.
- **Editable cells**: Double-click, press F2, or Enter to edit a cell. Changes are written back to the DataFrame.
- **Delete**: Select a row or column header and press Delete (or Backspace) to remove it.
- **Undo / Redo**: Ctrl+Z and Ctrl+Y (also Ctrl+R).
- **Deselect all**: Ctrl+D.

### What you should do here
Load your data, check column names, remove unwanted rows or columns, and correct any typing errors before running analyses.

---

## 5. Assumptions tab — detailed

### Controls
- **Alpha**: Significance level (default 0.05). Used for all residual checks and later tests.
- **Variables**: Multi-select list of numeric response columns.
- **Factor**: Single-select list. This is the factor whose one-way residuals are tested.
- **Run Full Assumption Check**: Starts the analysis.
- **Save Assumptions + Transformed Data**: Writes an Excel file with the results table and the chosen transformed columns.
- **Show Diagnostic Plots / Save Diagnostic Figure**: Residual histograms and Q-Q plots on the chosen scale.
- **Show Transformed Diagnostics / Save Transformed Diagnostic**: Same plots after transformation.

### What the check does (step by step)
1. Drops missing values for the variable + factor.
2. Computes one-way residuals (each observation minus its group mean).
3. Tests normality of those residuals:
   - Shapiro-Wilk if total n < 50
   - D’Agostino-Pearson otherwise
4. Tests homogeneity with the median-centered Levene test.  
   If any group has n < 3 the Levene p-value is reported as “Not tested” (not a failure).
5. Tries the transform ladder in order and **stops at the first scale** that meets the rule:
   - Raw
   - Natural log (or log1p / shifted log)
   - Square root
   - Arcsinh
6. Assigns a status: `both`, `normal_untested`, `normal_only`, `neither`, or insufficient data.

### Output table columns
- Variable  
- Fulfills Normality / Homogeneity  
- Residual normality p, Homogeneity p (Levene)  
- Residual skewness, Excess kurtosis  
- Chosen Transformation, Lambda  
- Transforms Evaluated  
- Min group size, Total n, Low power warning  
- **Required Analysis** (the sentence that tells you which test to use)  
- Interpretation Note  

---

## 6. ANOVA / KW tab — detailed

### Left-hand controls

**Dependent Variables**  
Multi-select list of responses to analyze.

**Factors**  
Multi-select list.  
If you select factors but do **not** use the Model Builder, the program automatically treats every selected factor as a main-effect term and produces an ANOVA table that includes all of them.

**Model Builder**
- **Add Mains**: Adds the currently selected factors as main-effect terms.
- **Add Inter.**: Adds the interaction of the selected factors.
- **Clear**: Empties the model list.
- Model list and Random list: You can mark terms as random (Ctrl+Click).  
  Only random intercepts are supported.

**Post-hoc & Mean Separation**
- ANOVA post-hoc: Tukey HSD, Bonferroni, Scheffé, Games-Howell, Dunnett, LSD, Šidák, None.
- Kruskal-Wallis post-hoc: Dunn (Bonferroni/Holm), Conover, Nemenyi, DSCF, Mann-Whitney, None.
- Factor for mean separation: The factor whose levels receive letters.
- Dunnett control: Required when Dunnett is chosen.

**Checkboxes**
- Force ANOVA: Overrides the automatic rule and stays in the ANOVA family (labeled).
- Force KW: Forces Kruskal-Wallis and drops the factorial model (labeled).
- Prefer model-adjusted contrasts (when available): Flags the Procedure line.
- Disable automatic scale search (force raw): Prevents the transform ladder; Procedure line shows “scale search: DISABLED”.

**Run Analysis / Clear**  
Starts the threaded analysis or clears the results panes.

### Right-hand results (Notebook)

**Notes tab**  
- Limitations banner (first time)  
- Variable name  
- **Procedure line** (example):  
  `Procedure: Type II ANOVA (Raw) — follows the Assumptions rule | scale search: active`  
- Test name, statistic, and p-values  
- When a selection-adjusted p-value exists it is shown as the **headline p**; the ordinary p is labeled conditional.  
- Short or full notes (controlled by the Short summary checkbox)

**ANOVA Table tab**  
Sortable Treeview.  
Contains one block per variable (Variable column added).  
Shows sources, degrees of freedom, sums of squares / mean squares, F (or Wald), and p-values.  
Copy button copies the visible table as tab-separated text.

**Means / Post-hoc tab**  
Sortable Treeview with group means, N, and compact letters (or other separation symbols).  
Also accumulates every variable.

**Details tab**  
Receives the long methodological notes when Short summary is checked, so the Notes tab stays short.

### Bottom buttons
- **Save results (Excel)**: One sheet per variable for the ANOVA table and the means table.
- **Export current view (.txt)**: Saves the Notes pane.
- **Copy notes**: Copies the Notes text.
- **Copy all tables**: Copies both Treeviews.
- **Copy selected variable tables**: Copies the current Treeview contents.
- **Run validation**: Runs the built-in self-tests and offers to save the summary.
- **Copy methods paragraph**: Copies a ready-to-paste methods sentence.
- **Short summary** (default on): Hides long notes (they move to the Details tab).

---

## 7. Decision logic (exact rules)

Residual checks always use **one-way residuals of the separation factor**.

| Status              | What it means                              | One-way result     | Multi-factor / block result      |
|---------------------|--------------------------------------------|--------------------|----------------------------------|
| both                | Normality and homogeneity both pass        | ANOVA              | ANOVA (model kept)               |
| normal_untested     | Normality passes, Levene could not be run  | ANOVA              | ANOVA                            |
| normal_only         | Normality passes, homogeneity fails        | Welch ANOVA        | HC3 or ART (if random term)      |
| neither / small     | Neither check passed                       | Kruskal-Wallis     | Aligned-rank ANOVA (ART)         |

- Force ANOVA or Force KW overrides the table above and is clearly labeled “does not follow the automatic rule”.
- A random term + a searched scale uses ART rather than a mixed model on the transformed scale.
- Selection-adjusted p-value (when computed) becomes the headline p-value.

---

## 8. Two-Sample Tests tab

- Choose Variable, Grouping factor, Group 1, Group 2.
- Check the tests you want (Independent t, Welch t, Mann-Whitney, Levene, etc.).
- Run Two-Sample Tests.
- Results appear in the text pane; you can save them.

---

## 9. Figures tab

- Select Variables and Factors.
- Choose error bar (SE or SD) and figure size.
- Draw produces a bar chart with error bars.
- Double-click the figure or use the edit fields to change titles and axis labels.
- Colors can be picked and applied.
- Save figure and (for two-sample) export data + chart to Excel.

---

## 10. Major functions (reference)

### Assumption and routing
- `choose_scale` — primary one-way residual checks and transform ladder.
- `assumption_row` — turns a decision into a results-table row.
- `analyze_variable` — full analysis for one response; returns the dictionary the GUI displays.
- `_procedure_sentence` — builds the single Procedure line.
- `_common_limitations` — text used in the banner and Details tab.

### Models and tests
- `classical_anova_table` — Type II ANOVA.
- `fit_mixed` — random-intercept mixed model.
- `art_tables` — Wobbrock aligned-rank transform.
- Kenward-Roger helpers — custom adjustment (used for the printed table and for permutation p-values).

### Post-hoc and letters
- `games_howell_pvalues`, `tukey_pvalues`, `dunn_pvalues`, `conover_pvalues`, `nemenyi_pvalues`, etc.
- `assign_compact_letters` — compact letter display from the non-significance graph.
- `holm_adjust`, `bonferroni_adjust`, `sidak_adjust`.

### GUI
- `EnhancedTool.run_anova` / `show_an` — run and display.
- `_fill_tree` / `_append_variable_table` — multi-variable Treeviews.
- `run_validation_gui`, `copy_methods_paragraph`, `copy_all_tables`, `export_anova_txt`.

---

## 11. How to report results

1. State the procedure that appears on the Procedure line.
2. If a selection-adjusted p-value is shown, report it as the primary p-value and mention that the ordinary p-value is conditional on the scale search.
3. Give the ANOVA (or ART / Welch) table.
4. Give the means and letters.
5. If you used a mixed model, note that it is random-intercept only and that the Kenward-Roger p-value should be cross-checked with lmerTest for journal submission.
6. Use the **Copy methods paragraph** button for a ready-made sentence.

---

## 12. Limitations (state these if you cite the tool)

- Transform search is selection-optimistic (mitigated by headline selection-adjusted p and the disable option).
- Mixed models support random intercepts only (custom Kenward-Roger).
- Pairwise tests are marginal on the separation factor by default.
- Custom KR and selection-adjusted paths are design-specific.
- Validation is internal (`--selftest` and the GUI button).

---

## 13. Troubleshooting

| Problem | Solution |
|---------|----------|
| “Select factors or build a model” | Select at least one factor or add terms in the Model Builder. |
| Treeview shows only one variable | Re-run; tables now accumulate. If still wrong, restart the program. |
| Long notes missing | Look in the Details tab or uncheck Short summary. |
| Program will not start | Run it in a terminal; it will list missing packages. Install them with pip. On Linux also install `python3-tk` if the GUI fails. |
| Mixed-model p-value questioned | Cross-check with R `lmerTest` as the output suggests. |

---

End of detailed manual.
