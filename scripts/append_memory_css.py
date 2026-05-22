"""Append Working Memory CSS to ui/style.css."""
import pathlib

CSS = """

/* =====================================================================
   Working Memory UI
   ===================================================================== */

.memory-section { padding: 0.875rem 1rem; border-bottom: 1px solid rgba(110,168,254,0.06); }
.memory-section:last-child { border-bottom: none; }
.memory-section--reset { padding: 0.75rem 1rem; display: flex; justify-content: center; }
.memory-section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem; }
.memory-section-title { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: rgba(225,229,235,0.4); }
.memory-section-badge { font-size: 0.6rem; font-family: 'JetBrains Mono', monospace; color: rgba(110,168,254,0.5); background: rgba(110,168,254,0.06); border: 1px solid rgba(110,168,254,0.1); border-radius: 4px; padding: 0.05rem 0.35rem; }
.memory-variable-list { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.6rem; }
.memory-variable-chip { display: flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.5rem; background: rgba(110,168,254,0.05); border: 1px solid rgba(110,168,254,0.1); border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; transition: border-color 0.15s ease; }
.memory-variable-chip:hover { border-color: rgba(110,168,254,0.22); }
.mem-var-key { color: #6ea8fe; font-weight: 500; }
.mem-var-sep { color: rgba(225,229,235,0.2); }
.mem-var-val { color: rgba(225,229,235,0.75); flex: 1; word-break: break-all; }
.memory-task-list { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.6rem; max-height: 180px; overflow-y: auto; }
.memory-task-row { display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.5rem; border-radius: 6px; border: 1px solid rgba(225,229,235,0.06); background: rgba(255,255,255,0.02); font-size: 0.75rem; transition: border-color 0.15s ease; }
.memory-task-row:hover { border-color: rgba(225,229,235,0.12); }
.task-pending { color: rgba(225,229,235,0.6); }
.task-in-progress { color: #fbbf24; border-color: rgba(251,191,36,0.12) !important; background: rgba(251,191,36,0.04) !important; }
.task-completed { color: rgba(225,229,235,0.25); text-decoration: line-through; }
.task-failed { color: rgba(248,113,113,0.6); border-color: rgba(248,113,113,0.12) !important; }
.mem-status-btn { flex-shrink: 0; width: 18px; height: 18px; border: none; background: transparent; cursor: pointer; color: inherit; font-size: 0.8rem; line-height: 1; border-radius: 4px; transition: background 0.15s ease; }
.mem-status-btn:hover { background: rgba(255,255,255,0.08); }
.mem-task-title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mem-prio-badge { flex-shrink: 0; font-size: 0.55rem; font-family: 'JetBrains Mono', monospace; padding: 0.05rem 0.3rem; border-radius: 3px; background: rgba(167,139,250,0.12); color: rgba(167,139,250,0.7); border: 1px solid rgba(167,139,250,0.15); }
.memory-obs-list { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.6rem; max-height: 150px; overflow-y: auto; }
.memory-add-row { display: flex; gap: 0.4rem; align-items: center; }
.memory-add-col { display: flex; flex-direction: column; gap: 0.4rem; }
.memory-input { background: rgba(30,36,51,0.8); border: 1px solid rgba(110,168,254,0.1); border-radius: 6px; padding: 0.3rem 0.5rem; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: rgba(225,229,235,0.85); transition: border-color 0.15s ease; }
.memory-input:focus { outline: none; border-color: rgba(110,168,254,0.3); }
.memory-input::placeholder { color: rgba(225,229,235,0.2); }
.memory-select { appearance: none; background: rgba(30,36,51,0.8); border: 1px solid rgba(110,168,254,0.1); border-radius: 6px; padding: 0.3rem 0.5rem; font-size: 0.7rem; font-family: inherit; color: rgba(225,229,235,0.8); cursor: pointer; }
.memory-select:focus { outline: none; border-color: rgba(110,168,254,0.3); }
.memory-add-btn { flex-shrink: 0; padding: 0.3rem 0.6rem; border-radius: 6px; border: 1px solid rgba(110,168,254,0.18); background: rgba(110,168,254,0.07); color: #6ea8fe; font-size: 0.68rem; font-family: inherit; cursor: pointer; white-space: nowrap; transition: all 0.15s ease; }
.memory-add-btn:hover { background: rgba(110,168,254,0.14); border-color: rgba(110,168,254,0.35); }
.mem-delete-btn { flex-shrink: 0; width: 16px; height: 16px; border: none; background: transparent; color: rgba(248,113,113,0.35); font-size: 0.85rem; line-height: 1; cursor: pointer; border-radius: 3px; display: flex; align-items: center; justify-content: center; transition: all 0.15s ease; margin-left: auto; }
.mem-delete-btn:hover { color: #f87171; background: rgba(248,113,113,0.1); }
.memory-reset-btn { padding: 0.4rem 1rem; border-radius: 6px; border: 1px solid rgba(248,113,113,0.15); background: rgba(248,113,113,0.04); color: rgba(248,113,113,0.5); font-size: 0.68rem; font-family: inherit; cursor: pointer; transition: all 0.15s ease; }
.memory-reset-btn:hover { background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.3); color: #f87171; }
"""

target = pathlib.Path("ui/style.css")
target.write_text(target.read_text(encoding="utf-8") + CSS, encoding="utf-8")
print("CSS appended OK")
