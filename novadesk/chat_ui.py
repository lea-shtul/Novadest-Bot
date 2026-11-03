"""NovaDesk chat UI (Streamlit).

Run from the project root:  streamlit run chat_ui.py

Calls the RAG pipeline directly (no HTTP), so the FastAPI server doesn't need to be running.
The history is for display only: each question is answered on its own.
"""

import logging

import streamlit as st

from app.rag.pipeline import ProviderUnavailableError, answer_question

logging.basicConfig(level=logging.INFO)  # show the pipeline's log lines (and errors) in the terminal
logger = logging.getLogger("chat_ui")

EXAMPLES = [
    "ה-VPN לא מתחבר מבית קפה",
    "יוצאים דפים לבנים מהמדפסת בקומה 3",
    "איך מזמינים מסך נוסף לעמדה?",
    "החשבון שלי ננעל, מה עושים?",
]
TYPE_LABELS = {"guide": "מדריך", "glossary": "מילון", "ticket": "טיקט"}
NOT_FOUND = "לא מצאתי מידע על זה במאגר הידע של נובה טק. כדאי לפנות לצוות IT בשלוחה 7700."
UNAVAILABLE = "השירות לא זמין כרגע, נסו שוב בעוד דקה"
GENERIC_ERROR = "משהו השתבש. נסו שוב, ואם זה חוזר, פנו לצוות IT בשלוחה 7700."

# Make the whole page right-to-left; code (paths like \\nova-files\drivers) stays left-to-right.
RTL_CSS = """
<style>
  .stApp { direction: rtl; }
  .stApp p, .stApp h1, .stApp li { text-align: right; }
  .stApp code, .stApp pre { direction: ltr; unicode-bidi: isolate; }
</style>
"""


def ask(question: str) -> dict:
    """Run the pipeline and turn the result (or error) into a message to display."""
    try:
        result = answer_question(question)
    except ProviderUnavailableError:
        return {"role": "assistant", "kind": "error", "text": UNAVAILABLE}
    except Exception:
        logger.exception("answer_question failed")  # the real error, in the terminal
        return {"role": "assistant", "kind": "error", "text": GENERIC_ERROR}

    if not result.answered:
        return {"role": "assistant", "kind": "not_found", "text": NOT_FOUND}
    return {"role": "assistant", "kind": "answer", "text": result.answer, "sources": result.sources}


def show(message: dict) -> None:
    """Draw the content of one message (call inside st.chat_message)."""
    kind = message.get("kind")
    if kind == "error":
        st.error(message["text"])
    elif kind == "not_found":
        st.info(message["text"])
    else:
        st.markdown(message["text"])
    if kind == "answer":
        with st.expander("קטעים שנבדקו"):
            for source in message["sources"]:
                st.markdown(source_line(source))


def source_line(source) -> str:
    parts = [f"**{TYPE_LABELS.get(source.type, source.type)}**", source.title]
    if source.type == "ticket" and source.resolved_by_name:
        parts.append(f"נפתר ע״י {source.resolved_by_name}")
    elif source.section:
        parts.append(source.section)
    parts.append(f"ציון {source.score:.2f}")
    return "- " + " · ".join(parts)


def new_chat() -> None:
    st.session_state.history = []


def send_example(question: str) -> None:
    st.session_state.pending = question  # picked up below, on the rerun the click triggers


# --- The page ---

st.set_page_config(page_title="NovaDesk", page_icon="💬")
st.markdown(RTL_CSS, unsafe_allow_html=True)

st.session_state.setdefault("history", [])

st.title("NovaDesk")
st.caption("הבוט של צוות ה-IT בנובה טק. שאלו שאלה, וקבלו תשובה מתוך המדריכים, המילון והטיקטים שנפתרו.")
st.button("שיחה חדשה", on_click=new_chat)

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        show(message)

question = st.chat_input("כתבו שאלה...") or st.session_state.pop("pending", None)

if not st.session_state.history and not question:
    st.write("במה אפשר לעזור? לדוגמה:")
    for example in EXAMPLES:
        st.button(example, on_click=send_example, args=(example,))

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("חושב..."):
            reply = ask(question)
        show(reply)
    st.session_state.history += [{"role": "user", "text": question}, reply]
