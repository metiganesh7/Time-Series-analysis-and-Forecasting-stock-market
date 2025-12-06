# main.py - Diagnostic (drop this in, commit & redeploy)
import sys, os, traceback
import streamlit as st
st.set_page_config(page_title="Diagnostics", layout="wide")

st.title("🔍 App Diagnostic — shows imports, files, and errors")

# 1) Python & paths
st.header("Environment")
st.write("Python:", sys.version.splitlines()[0])
st.write("Working dir:", os.getcwd())
st.write("Files in repo root (top 50):")
files = os.listdir(".")
files.sort()
st.write(files[:50])

# 2) Show scripts folder contents (if exists)
scripts_path = os.path.join(os.getcwd(), "scripts")
if os.path.isdir(scripts_path):
    st.write("scripts/ exists. Listing:")
    st.write(sorted(os.listdir(scripts_path)))
else:
    st.error("No scripts/ folder found at repo root. That will break the app if models are imported.")

# 3) Check main file name
st.write("Streamlit main file currently deployed as: main.py (this diagnostic file).")

# 4) Try importing optional heavy libs and show results (safe)
st.header("Check optional libraries")
libs = ["plotly", "prophet", "yfinance", "statsmodels", "tensorflow", "sklearn"]
lib_status = {}
for lib in libs:
    try:
        __import__(lib)
        lib_status[lib] = "OK"
    except Exception as e:
        lib_status[lib] = f"ERROR: {e.__class__.__name__}: {str(e)}"
st.json(lib_status)

# 5) Try to import your scripts and show detailed tracebacks (safe)
st.header("Attempt to import scripts/*.py (captured errors below)")
import_errors = []
if os.path.isdir(scripts_path):
    for fname in sorted(os.listdir(scripts_path)):
        if fname.endswith(".py"):
            mod = fname[:-3]
            st.write(f"--- trying import: scripts.{mod} ---")
            try:
                # remove any stale module
                if f"scripts.{mod}" in sys.modules:
                    del sys.modules[f"scripts.{mod}"]
                # attempt import
                module = __import__(f"scripts.{mod}", fromlist=["*"])
                st.success(f"Imported scripts.{mod} OK — attributes: {sorted([a for a in dir(module) if not a.startswith('_')])[:50]}")
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"Failed to import scripts.{mod}")
                st.code(tb)
                import_errors.append((mod, str(e)))
else:
    st.warning("No scripts folder to import from.")

# 6) Quick sanity test UI
st.header("Quick UI test")
st.write("If you see this text and the checks above, Streamlit rendering is working.")
st.button("This button works — click me")

# 7) Next steps text
st.markdown("""
**Next steps**
- If any `scripts.*` import failed above, copy the error shown and paste it here (or fix the file).
- If a library shows ERROR above (like `prophet` or `plotly`), add it to requirements.txt or remove references.
- If `scripts/` is missing, re-add it to your repo.
""")
