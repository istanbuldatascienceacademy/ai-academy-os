
# app.py — AI Academy OS | Streamlit Multi-Agent Arayüzü

import streamlit as st
import openai
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# ─────────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Academy OS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.agent-badge {
    display:inline-block;padding:3px 10px;border-radius:12px;
    font-size:12px;font-weight:600;margin-bottom:8px;
}
.badge-student  { background:#dbeafe;color:#1e40af; }
.badge-syllabus { background:#dcfce7;color:#166534; }
.badge-exam     { background:#fef9c3;color:#854d0e; }
.badge-content  { background:#f3e8ff;color:#6b21a8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# AGENT ALTYAPISI
# ─────────────────────────────────────────────────────────────────
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    task_type: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time()*1000)}")

@dataclass
class AgentResponse:
    agent_name: str
    task_type: str
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

def llm_call(messages, model="gpt-4o-mini", temperature=0.7,
             max_tokens=1500, json_mode=False):
    client = openai.OpenAI(api_key=st.session_state.api_key)
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content.strip()

class BaseAgent:
    def __init__(self, name, role, system_prompt, model="gpt-4o-mini"):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.history: List[Dict] = []
        self.task_count = 0

    def _call(self, user_msg, temperature=0.7, max_tokens=1500, json_mode=False):
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs += self.history[-6:]
        msgs.append({"role": "user", "content": user_msg})
        result = llm_call(msgs, model=self.model, temperature=temperature,
                          max_tokens=max_tokens, json_mode=json_mode)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": result})
        self.task_count += 1
        return result

    def reset(self):
        self.history = []

COURSE_DATABASE = {
    "AI101": {
        "ad": "Yapay Zekaya Giriş", "kredi": 3, "donem": "1. Dönem",
        "aciklama": "AI tarihi, ML temelleri, uygulama alanları",
        "konular": ["AI tarihi", "ML türleri", "Derin öğrenme", "Etik", "Uygulamalar"],
        "on_kosul": [], "ogretim_elemani": "Dr. Ahmet Yılmaz",
        "degerlendirime": {"vize": 30, "proje": 30, "final": 40}
    },
    "DS201": {
        "ad": "Veri Bilimi ve Analitik", "kredi": 4, "donem": "2. Dönem",
        "aciklama": "Python ile veri analizi, görselleştirme, istatistik",
        "konular": ["Python temelleri", "Pandas/NumPy", "Görselleştirme", "İstatistik", "ML pipeline"],
        "on_kosul": ["AI101"], "ogretim_elemani": "Dr. Zeynep Arslan",
        "degerlendirime": {"vize": 25, "lab": 35, "final": 40}
    },
    "LLM301": {
        "ad": "Büyük Dil Modelleri", "kredi": 3, "donem": "3. Dönem",
        "aciklama": "Transformer mimarisi, GPT, fine-tuning, RAG sistemleri",
        "konular": ["Transformer", "Pre-training", "Fine-tuning", "Prompt Eng.", "RAG", "Agents"],
        "on_kosul": ["AI101", "DS201"], "ogretim_elemani": "Zekeriya Besiroglu",
        "degerlendirime": {"vize": 20, "proje": 40, "final": 40}
    },
    "FIN401": {
        "ad": "Fintech & AI Uygulamaları", "kredi": 3, "donem": "4. Dönem",
        "aciklama": "Bankacılık AI'ı, risk modelleri, regtech",
        "konular": ["Fintech", "Kredi riski", "Fraud tespiti", "RegTech", "Open Banking"],
        "on_kosul": ["DS201", "LLM301"], "ogretim_elemani": "Zekeriya Besiroglu",
        "degerlendirime": {"vize": 30, "proje": 30, "final": 40}
    },
}

class StudentAssistantAgent(BaseAgent):
    SYSTEM = """Sen Istanbul Data Science Academy'nin deneyimli akademik danışmanısın.
Öğrencilere ders seçimi, kariyer planlaması, staj ve motivasyon konularında yardım ediyorsun.
Yanıtlarında: empati kur, somut öneriler ver, Türkçe yaz."""
    def __init__(self):
        super().__init__("StudentAssistant", "Akademik Danışman", self.SYSTEM)
    def process(self, message):
        answer = self._call(message.payload.get("query",""), temperature=0.7, max_tokens=800)
        return AgentResponse(agent_name=self.name, task_type="student_query",
                             result={"answer": answer})

class SyllabusAgent(BaseAgent):
    def __init__(self, course_db):
        db_str = json.dumps(course_db, ensure_ascii=False, indent=2)
        system = f"Sen müfredat danışmanısın. DERS VERİTABANI:\n{db_str}\nSadece veritabanındaki bilgileri kullan. Türkçe yaz."
        super().__init__("SyllabusBot", "Müfredat Danışmanı", system)
    def process(self, message):
        answer = self._call(message.payload.get("query",""), temperature=0.3, max_tokens=700)
        return AgentResponse(agent_name=self.name, task_type="syllabus",
                             result={"answer": answer})

class ExamGeneratorAgent(BaseAgent):
    SYSTEM = "Sen akademik sınav tasarımcısısın. Bloom taksonomisine uygun sorular üret. SADECE JSON döndür."
    def __init__(self):
        super().__init__("ExamGenerator", "Sınav Tasarımcısı", self.SYSTEM)
    def process(self, message):
        p = message.payload
        prompt = f"""Konu: {p.get('konu','')}, Ders: {p.get('ders_kodu','')},
Soru sayısı: {p.get('sayi',4)}, Zorluk: {p.get('zorluk','orta')}
JSON: {{"sinav_basligi":"...","sorular":[{{"soru":"...","secenekler":["A)...","B)...","C)...","D)..."],"dogru_yanit":"A","aciklama":"...","zorluk":"orta","puan":10}}],"toplam_puan":100,"sure_dakika":60}}"""
        raw = self._call(prompt, temperature=0.5, max_tokens=2000, json_mode=True)
        try:
            return AgentResponse(agent_name=self.name, task_type="exam",
                                 result=json.loads(raw))
        except:
            return AgentResponse(agent_name=self.name, task_type="exam",
                                 result=None, success=False, error="JSON hatası")

class ContentCreatorAgent(BaseAgent):
    SYSTEM = "Sen akademik içerik yazarısın. Yapılandırılmış, yüksek kaliteli Türkçe içerik üret."
    TIPLER = {
        "ders_notu": "Kapsamlı ders notu (giriş, konular, örnekler, özet)",
        "vaka_calismasi": "Gerçek dünya vaka analizi",
        "ozet": "Hızlı referans özet kartı",
        "proje_tanimi": "Dönem projesi tanımı ve değerlendirme kriterleri",
    }
    def __init__(self):
        super().__init__("ContentCreator", "İçerik Yazarı", self.SYSTEM)
    def process(self, message):
        p = message.payload
        tip = p.get("tip", "ders_notu")
        ek_str = f"Ek talimat: {p['ek_talimat']}" if p.get('ek_talimat') else ''
        prompt = f"İcerik turu: {self.TIPLER.get(tip, tip)}\nKonu: {p.get('konu','')}\nHedef kitle: {p.get('hedef_kitle','lisans ogrencisi')}\n{ek_str}"
        content = self._call(prompt, temperature=0.7, max_tokens=1200)
        return AgentResponse(agent_name=self.name, task_type="content",
                             result={"icerik": content, "tip": tip})

class OrchestratorAgent:
    ROUTER = """İsteği analiz et. SADECE JSON:
{"task_type":"student_query|syllabus|exam|content","confidence":0.0-1.0,"reasoning":"kisa"}"""
    def __init__(self):
        self.agents = {
            "student_query": StudentAssistantAgent(),
            "syllabus": SyllabusAgent(COURSE_DATABASE),
            "exam": ExamGeneratorAgent(),
            "content": ContentCreatorAgent(),
        }
        self.stats = {k: 0 for k in self.agents}
        self.total = 0

    def _route(self, req):
        r = llm_call([{"role":"system","content":self.ROUTER},
                      {"role":"user","content":f"İstek: {req}"}],
                     temperature=0.0, max_tokens=120, json_mode=True)
        try: return json.loads(r)
        except: return {"task_type":"student_query","confidence":0.5,"reasoning":"fallback"}

    def _build_payload(self, req, task_type):
        if task_type == "exam":
            try:
                return json.loads(llm_call(
                    [{"role":"user","content":f"'{req}' isteğinden: JSON {{\"konu\":\"\",\"ders_kodu\":\"\",\"sayi\":4,\"zorluk\":\"orta\"}}"}],
                    temperature=0.0, max_tokens=150, json_mode=True))
            except: return {"konu": req, "sayi": 4, "zorluk": "orta"}
        elif task_type == "content":
            try:
                return json.loads(llm_call(
                    [{"role":"user","content":f"'{req}' isteğinden: JSON {{\"konu\":\"\",\"tip\":\"ders_notu\",\"hedef_kitle\":\"lisans öğrencisi\",\"ek_talimat\":\"\"}}"}],
                    temperature=0.0, max_tokens=150, json_mode=True))
            except: return {"konu": req, "tip": "ders_notu"}
        return {"query": req}

    def run(self, req):
        self.total += 1
        t0 = time.time()
        route = self._route(req)
        task_type = route.get("task_type", "student_query")
        payload = self._build_payload(req, task_type)
        agent = self.agents.get(task_type, self.agents["student_query"])
        resp = agent.process(AgentMessage("user", agent.name, task_type, payload))
        self.stats[task_type] = self.stats.get(task_type, 0) + 1
        resp.metadata.update({"routed_to": task_type,
                               "confidence": route.get("confidence", 0),
                               "elapsed": round(time.time()-t0, 2)})
        return resp

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "orch" not in st.session_state:
    st.session_state.orch = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 AI Academy OS")
    st.caption("Istanbul Data Science Academy")
    st.divider()

    st.subheader("🔑 API Ayarları")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=st.session_state.api_key, placeholder="sk-...")
    if api_key:
        st.session_state.api_key = api_key

    st.divider()
    st.subheader("🤖 Ajanlar")
    badges = [
        ("👨‍🎓 Öğrenci Asistanı", "badge-student", "Kariyer & rehberlik"),
        ("📚 Müfredat Botu",     "badge-syllabus", "Ders & program"),
        ("📝 Sınav Üretici",     "badge-exam",     "Bloom taksonomisi"),
        ("✍️ İçerik Üretici",    "badge-content",  "Ders notu & vaka"),
    ]
    for name, cls, desc in badges:
        st.markdown(f'<span class="agent-badge {cls}">{name}</span>  <small>{desc}</small>',
                    unsafe_allow_html=True)

    st.divider()
    if st.session_state.orch:
        st.subheader("📊 İstatistik")
        o = st.session_state.orch
        st.metric("Toplam İstek", o.total)
        cols = st.columns(2)
        icons = {"student_query":"👨‍🎓","syllabus":"📚","exam":"📝","content":"✍️"}
        for i,(k,v) in enumerate(o.stats.items()):
            with cols[i%2]:
                st.metric(f"{icons.get(k,'')} {k[:7]}", v)

    st.divider()
    if st.button("🔄 Temizle", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.orch:
            for a in st.session_state.orch.agents.values():
                a.reset()
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# ANA ALAN
# ─────────────────────────────────────────────────────────────────
st.title("🎓 AI Academy OS")
st.caption("Multi-Agent Üniversite Asistanı · Zekeriya Besiroglu | IDSA")

if not st.session_state.api_key:
    st.warning("⬅️ Sol panelden OpenAI API key'inizi girin.")
    st.info("""**Ne yapar?**
- 👨‍🎓 Öğrenci rehberliği & kariyer danışmanlığı
- 📚 Müfredat chatbot & ders bilgisi
- 📝 Otomatik sınav sorusu üretimi
- ✍️ Ders notu, vaka, proje tanımı üretimi
- 🔄 Tek tıkla tam ders paketi pipeline""")
    st.stop()

if st.session_state.orch is None:
    with st.spinner("Ajanlar başlatılıyor..."):
        st.session_state.orch = OrchestratorAgent()

orch = st.session_state.orch

# ─── SEKMELER ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Sohbet", "📝 Sınav Üret", "✍️ İçerik Üret", "🔄 Pipeline", "📚 Müfredat"
])

AGENT_ICONS  = {"student_query":"🟦","syllabus":"🟩","exam":"🟨","content":"🟪"}
AGENT_LABELS = {"student_query":"Öğrenci Asistanı","syllabus":"Müfredat Botu",
                "exam":"Sınav Üretici","content":"İçerik Üretici"}

# ── TAB 1: SOHBET ──────────────────────────────────────────────
with tab1:
    st.subheader("Akıllı Sohbet")
    st.caption("Sistem isteği analiz edip otomatik olarak doğru ajana yönlendirir.")

    with st.expander("💡 Örnek sorular"):
        examples = [
            "Veri bilimi kariyerine nasıl başlarım?",
            "LLM301 dersinin ön koşulları neler?",
            "RAG hakkında kısa ders notu yaz",
            "Fine-tuning konusunda 3 soru üret",
            "Staj başvurusu için portfolyo nasıl hazırlanır?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i%2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":ex})
                with st.spinner("Yanıt üretiliyor..."):
                    resp = orch.run(ex)
                answer = ""
                if resp.success and isinstance(resp.result, dict):
                    answer = resp.result.get("answer") or resp.result.get("icerik","")
                st.session_state.chat_history.append({
                    "role":"assistant","content":answer,
                    "agent":resp.metadata.get("routed_to"),
                    "elapsed":resp.metadata.get("elapsed"),
                })
                st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("agent"):
                a = msg["agent"]
                elapsed_str = f" · {msg['elapsed']}s" if msg.get('elapsed') else ''
                st.caption(f"{AGENT_ICONS.get(a,'🤖')} **{AGENT_LABELS.get(a,a)}**{elapsed_str}")
            st.markdown(msg["content"])

    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Yanıt üretiliyor..."):
                resp = orch.run(prompt)
            answer = ""
            if resp.success and isinstance(resp.result, dict):
                answer = (resp.result.get("answer") or resp.result.get("icerik") or
                          json.dumps(resp.result, ensure_ascii=False, indent=2))
            a = resp.metadata.get("routed_to","")
            st.caption(f"{AGENT_ICONS.get(a,'🤖')} **{AGENT_LABELS.get(a,a)}**"
                       f" · {resp.metadata.get('elapsed','')}s"
                       f" · güven: {resp.metadata.get('confidence',0):.0%}")
            st.markdown(answer)
            st.session_state.chat_history.append({
                "role":"assistant","content":answer,
                "agent":a,"elapsed":resp.metadata.get("elapsed"),
            })

# ── TAB 2: SINAV ÜRET ──────────────────────────────────────────
with tab2:
    st.subheader("📝 Otomatik Sınav Üretici")
    col1, col2 = st.columns([2,1])
    with col1:
        konu = st.text_input("Konu", placeholder="ör: Transformer mimarisi")
        ders_kodu = st.selectbox("Ders", ["LLM301","AI101","DS201","FIN401","Genel"])
    with col2:
        sayi = st.number_input("Soru sayısı", 1, 10, 4)
        zorluk = st.select_slider("Zorluk", ["kolay","orta","zor"], value="orta")

    if st.button("🚀 Sınav Üret", type="primary", use_container_width=True):
        if not konu:
            st.warning("Konu girin.")
        else:
            with st.spinner("Sorular üretiliyor..."):
                resp = orch.agents["exam"].process(
                    AgentMessage("user","ExamGenerator","exam",
                                 {"konu":konu,"ders_kodu":ders_kodu,"sayi":sayi,"zorluk":zorluk}))
            if resp.success and resp.result:
                data = resp.result
                st.success(f"✅ {len(data.get('sorular',[]))} soru hazır!")
                c1,c2,c3 = st.columns(3)
                c1.metric("Toplam Puan", data.get("toplam_puan",100))
                c2.metric("Süre", f"{data.get('sure_dakika',60)} dk")
                c3.metric("Soru", len(data.get("sorular",[])))
                for i, s in enumerate(data.get("sorular",[]),1):
                    with st.expander(f"Soru {i} · {s.get('puan',10)}p · {s.get('zorluk','').upper()}", expanded=i==1):
                        st.markdown(f"**{s['soru']}**")
                        for opt in s.get("secenekler",[]):
                            prefix = "✅ " if opt.startswith(s.get("dogru_yanit","?")+")" ) else "　 "
                            st.markdown(f"{prefix}`{opt}`")
                        st.info(f"💡 {s.get('aciklama','')}")
                st.download_button("⬇️ JSON İndir",
                                   json.dumps(data, ensure_ascii=False, indent=2),
                                   file_name=f"sinav_{konu[:20]}.json",
                                   mime="application/json")
            else:
                st.error(f"Hata: {resp.error}")

# ── TAB 3: İÇERİK ÜRET ─────────────────────────────────────────
with tab3:
    st.subheader("✍️ Akademik İçerik Üretici")
    tip_map = {"📖 Ders Notu":"ders_notu","🏢 Vaka Çalışması":"vaka_calismasi",
               "📋 Özet Kartı":"ozet","🎯 Proje Tanımı":"proje_tanimi"}
    col1, col2 = st.columns(2)
    with col1:
        icerik_konu = st.text_input("Konu", key="ik", placeholder="ör: RAG sistemleri")
        icerik_tip  = st.selectbox("Tür", list(tip_map.keys()))
    with col2:
        hedef = st.selectbox("Hedef Kitle",
                             ["lisans öğrencisi","yüksek lisans öğrencisi",
                              "profesyonel","C-level yönetici"])
        ek = st.text_area("Ek Talimat", height=100,
                          placeholder="ör: Fintech örnekleri kullan")

    if st.button("✨ İçerik Üret", type="primary", use_container_width=True):
        if not icerik_konu:
            st.warning("Konu girin.")
        else:
            with st.spinner("İçerik üretiliyor..."):
                resp = orch.agents["content"].process(
                    AgentMessage("user","ContentCreator","content",
                                 {"konu":icerik_konu,"tip":tip_map[icerik_tip],
                                  "hedef_kitle":hedef,"ek_talimat":ek or ""}))
            if resp.success and resp.result:
                content_text = resp.result.get("icerik","")
                st.success("✅ İçerik hazır!")
                st.markdown("---")
                st.markdown(content_text)
                st.download_button("⬇️ Markdown İndir", content_text,
                                   file_name=f"{icerik_konu[:20]}.md", mime="text/markdown")

# ── TAB 4: PIPELINE ─────────────────────────────────────────────
with tab4:
    st.subheader("🔄 Ders Paketi Pipeline")
    st.caption("Tek tıkla: ders notu + sınav soruları + öğrenci rehberliği")
    col1, col2 = st.columns(2)
    with col1:
        pipe_konu = st.text_input("Konu", key="pk", placeholder="ör: Hallucination ve RAG")
    with col2:
        pipe_ders = st.selectbox("Ders", ["LLM301","AI101","DS201","FIN401"], key="pd")

    if st.button("🚀 Pipeline Çalıştır", type="primary", use_container_width=True):
        if not pipe_konu:
            st.warning("Konu girin.")
        else:
            prog = st.progress(0, "Başlatılıyor...")
            prog.progress(10, "Adım 1/3: Ders notu...")
            r_note = orch.agents["content"].process(
                AgentMessage("orch","ContentCreator","content",
                             {"konu":pipe_konu,"tip":"ders_notu",
                              "hedef_kitle":f"{pipe_ders} öğrencileri"}))
            prog.progress(45, "Adım 2/3: Sınav soruları...")
            r_exam = orch.agents["exam"].process(
                AgentMessage("orch","ExamGenerator","exam",
                             {"konu":pipe_konu,"sayi":3,"zorluk":"orta","ders_kodu":pipe_ders}))
            prog.progress(80, "Adım 3/3: Rehberlik...")
            r_guide = orch.agents["student_query"].process(
                AgentMessage("orch","StudentAssistant","student_query",
                             {"query":f"{pipe_konu} konusunu adım adım nasıl öğrenirim?"}))
            prog.progress(100, "✅ Tamamlandı!")
            time.sleep(0.4)
            prog.empty()

            st.success(f"✅ '{pipe_konu}' paketi hazır!")
            t1, t2, t3 = st.tabs(["📖 Ders Notu","📝 Sınav","🎓 Rehberlik"])
            with t1:
                if r_note.success:
                    st.markdown(r_note.result.get("icerik",""))
            with t2:
                if r_exam.success and r_exam.result:
                    for i, s in enumerate(r_exam.result.get("sorular",[]),1):
                        with st.expander(f"Soru {i}"):
                            st.markdown(f"**{s['soru']}**")
                            for opt in s.get("secenekler",[]):
                                pfx = "✅ " if opt.startswith(s.get("dogru_yanit","?")+")" ) else "　 "
                                st.markdown(f"{pfx}{opt}")
                            st.info(f"💡 {s.get('aciklama','')}")
            with t3:
                if r_guide.success:
                    st.markdown(r_guide.result.get("answer",""))

            export = f"# {pipe_konu}\n\n## Ders Notu\n"
            if r_note.success:  export += r_note.result.get("icerik","") + "\n\n"
            if r_guide.success: export += "## Rehberlik\n" + r_guide.result.get("answer","")
            st.download_button("⬇️ Paketi İndir", export,
                               file_name=f"paket_{pipe_konu[:20]}.md", mime="text/markdown")

# ── TAB 5: MÜFREDAT ─────────────────────────────────────────────
with tab5:
    st.subheader("📚 Müfredat & Ders Kataloğu")
    col1, col2 = st.columns([3,1])
    with col1:
        sq = st.text_input("Müfredat sorusu", placeholder="ör: FIN401 ön koşulları?")
    with col2:
        df = st.selectbox("Filtre", ["Tümü"] + list(COURSE_DATABASE.keys()))

    if st.button("🔍 Sorgula", type="primary"):
        if sq:
            with st.spinner("Aranıyor..."):
                payload = {"query": sq}
                if df != "Tümü": payload["ders_kodu"] = df
                resp = orch.agents["syllabus"].process(
                    AgentMessage("user","SyllabusBot","syllabus", payload))
            if resp.success:
                st.markdown(resp.result.get("answer",""))

    st.divider()
    st.subheader("Ders Kataloğu")
    for kod, ders in COURSE_DATABASE.items():
        with st.expander(f"**{kod}** — {ders['ad']}  ·  {ders['donem']}  ·  {ders['kredi']} kredi"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Açıklama:** {ders['aciklama']}")
                st.markdown(f"**Öğretim Üyesi:** {ders['ogretim_elemani']}")
                st.markdown(f"**Ön Koşul:** {', '.join(ders['on_kosul']) or 'Yok'}")
            with c2:
                st.markdown("**Konular:**")
                for k in ders["konular"]: st.markdown(f"• {k}")
                st.markdown("**Değerlendirme:**")
                for tip, puan in ders["degerlendirime"].items():
                    st.progress(puan/100, text=f"{tip}: %{puan}")
