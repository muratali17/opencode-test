"""
AutoGluon + Gradio: Uçtan uca Makine Öğrenmesi Web Uygulaması
=============================================================

Hedef kitle: makine öğrenmesi bilmeyen kullanıcılar. Açıklamaları okuyup
"Modeli Eğit" düğmesine basmaları yeterli.

Akış:
  1) CSV veri yükleme + hedef değişken seçimi
  2) Model grubu seçimi (AutoGluon resmi model anahtarları + yalın açıklama)
  3) TabularPredictor.fit() -> leaderboard görselleştirme

Çalıştırma:  python ml_app.py
Gereksinimler: gradio, autogluon.tabular, pandas
"""

import os
import time
import uuid
import logging
import traceback
import importlib.util

import pandas as pd
import gradio as gr

from autogluon.tabular import TabularPredictor

# ---------------------------------------------------------------------------
# Loglama: Eğitim sürecini ve hataları konsola yaz
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml-app")

WORKDIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(WORKDIR, "autogluon_models")  # tüm eğitimlerin üst dizini


# ---------------------------------------------------------------------------
# MODELLER (küratörlü seçim)
# ---------------------------------------------------------------------------
# AutoGluon'da ~40 model vardır; bu uygulama yalnızca tabular (yapısal) veriyle
# uğraştığı ve hedef kitle ML bilmediği için yalnızca "tabular'da kanıtlanmış"
# modelleri seçtik. Elenen gruplar ve nedenleri:
#   - Derin öğrenme (NN_TORCH, FASTAI, REALMLP, TABM): tabular veride ağaç
#     modellere genelde kaybeder; torch/fastai/pytabkit ~2GB kurulum ve uzun
#     eğitim süresi ister. Getiri/maliyet oranı kötü.
#   - Ön eğitimli / foundation modeller (MITRA, TABICL, TABPFN-3, NORI, ...):
#     model ağırlıklarını internetten indirir, bazıları ticari lisans ister
#     (TABPFN-3) ve NORI yalnızca regresyon çalışır. Production için riskli.
#   - DUMMY, GBM_PREP, ENS_WEIGHTED / SIMPLE_ENS_WEIGHTED: dahili / otomatik.
#   - AG_TEXT_NN / AG_IMAGE_NN / AG_AUTOMM: metin-görsel sütunları gerekir.
#   - TABPFNMIX, TABPFN-2.6, REALTABPFN-V2, TABDPT, FT_TRANSFORMER: deneysel,
#     yeni sürümle değiştirilmiş veya ağır.

# key -> (görünen ad, tek cümlelik yalın açıklama, zorunlu paket)
MODEL_META = {
    "GBM":          ("LightGBM",      "Hızlı ve dengeli bir ağaç modeli; hemen hemen her veride iyi çalışır.", "lightgbm"),
    "CAT":          ("CatBoost",      "Kategorik (etiketli) sütunlarda ve eksik veride çok başarılıdır.", "catboost"),
    "KNN":          ("KNeighbors",    "En benzer satırlara bakarak tahmin yapar; basit ama çok büyük veride yavaş.", "sklearn"),
    "XGB":          ("XGBoost",       "Güçlü ve popüler ağaç modeli; eksik ve karışık veride başarılı.", "xgboost"),
    "RF":           ("RandomForest",  "Birçok ağacın ortalamasını alır; sağlamdır, aşırı öğrenmeye dirençlidir.", "sklearn"),
    "XT":           ("ExtraTrees",    "Random Forest'a benzer, eğitimi biraz daha hızlıdır.", "sklearn"),
    "LR":           ("Linear",        "Basit ve çok anlaşılır doğrusal model; doğrusal ilişkilerde iyidir.", "sklearn"),
    "EBM":          ("EBM",           "Cam-kutu model; tahminin nedenlerini adım adım açıklar.", "interpret"),
    "IM_RULEFIT":   ("RuleFit",       "Ağaç + doğrusal karışımı; anlaşılır kurallar üretir.", "rulefit"),
    "IM_GREEDYTREE":("GreedyTree",    "Kısa, yorumlanabilir karar ağacı.", "sklearn"),
}

# Her modelin hangi Python paketine ihtiyaç duyduğu -> kurulu mu diye kontrol edilir
_MODEL_DEPENDENCY = {k: meta[2] for k, meta in MODEL_META.items()}


def _is_installed(module_name: str) -> bool:
    """Modül kurulu mu diye bakar (find_spec ile)."""
    if module_name == "sklearn":
        return importlib.util.find_spec("sklearn") is not None
    return importlib.util.find_spec(module_name) is not None


def _installed_models(model_keys):
    """Verilen anahtar listesinden kurulu olanları döndürür."""
    return [k for k in model_keys if _is_installed(_MODEL_DEPENDENCY.get(k, ""))]


# ---------------------------------------------------------------------------
# MODEL GRUPLARI
# ---------------------------------------------------------------------------
# Her grup: yalın Türkçe açıklama + AutoGluon model anahtarları.
# auto=True ise AutoGluon kendi en iyi seçimini yapar (presets kullanılır).
GROUPS = [
    {
        "id": "hizli",
        "label": "⚡ Hızlı ve Hafif",
        "desc": (
            "Eğitim saniyeler içinde biter ve yine de iyi sonuç verir. "
            "Hızlı bir fikir edinmek veya küçük-orta boy veri için idealdir."
        ),
        "models": ["GBM", "CAT", "KNN"],
    },
    {
        "id": "agac",
        "label": "🌳 Ağaç Modelleri",
        "desc": (
            "Eksik hücreleri ve sayı + metin karışık sütunları otomatik yönetir; "
            "genelde en sağlam ve başarılı sonuçları üretir. Çoğu senaryo için güvenli seçimdir."
        ),
        "models": ["XGB", "RF", "XT"],
    },
    {
        "id": "buyuk_veri",
        "label": "📈 Büyük Veri",
        "desc": (
            "Milyonlarca satırda bile verimli çalışan bellek dostu modeller. "
            "Veri setiniz çok büyükse bu grubu seçin."
        ),
        "models": ["GBM", "XGB", "CAT", "LR"],
    },
    {
        "id": "yorumlanabilir",
        "label": "🔍 Açıklanabilir Modeller",
        "desc": (
            "Kara kutu değildir; modelin 'neden böyle tahmin ettiğini' anlayabilirsiniz. "
            "Güven ve şeffaflık istediğinizde, örneğin raporlama veya yönetim sunumu için seçin."
        ),
        "models": ["LR", "EBM", "IM_RULEFIT", "IM_GREEDYTREE"],
    },
    {
        "id": "otomatik",
        "label": "🤖 AutoGluon Otomatik (En İyi Sonuç)",
        "desc": (
            "AutoGluon sizin için en iyi modelleri seçip birleştirir. "
            "Ne yapacağınızdan emin değilseniz en doğru tercih budur; eğitim uzun sürebilir."
        ),
        "models": [],
        "auto": True,
    },
]

GROUP_BY_LABEL = {g["label"]: g for g in GROUPS}
STRATEGY_LABELS = [g["label"] for g in GROUPS]


def group_hyperparameters(group: dict):
    """Grup -> TabularPredictor.fit() için (hyperparameters, presets) döndürür."""
    if group.get("auto"):
        return None, "best"          # AutoGluon kendi en iyi seçimini yapsın
    # Sadece kurulu modelleri çalıştır; eksik paketler hata üretmesin
    installed = _installed_models(group["models"])
    hp = {k: {} for k in installed}
    return hp if hp else None, None  # kurulu model yoksa AutoGluon karar versin


def group_status_text(group: dict) -> str:
    """Gruptaki kurulu/eksik modelleri yalın şekilde özetler."""
    if group.get("auto"):
        return "AutoGluon, kurulu tüm modelleri değerlendirip en iyisini seçer."
    active = _installed_models(group["models"])
    names = [MODEL_META[k][0] for k in active]
    if not active:
        return "Bu gruptaki modeller için gerekli paketler kurulu değil. Aşağıdaki kurulumu çalıştırın: `pip install " + " ".join(
            set(_MODEL_DEPENDENCY.get(k, "") for k in group["models"])) + "`"
    missing = [k for k in group["models"] if k not in active]
    text = "Bu grupta eğitilecek modeller: **" + ", ".join(names) + "**"
    if missing:
        text += "  \nKurulu olmayan (atlanacak): " + ", ".join(MODEL_META[k][0] for k in missing)
        text += "  \nKurulum için: `pip install " + " ".join(sorted(set(_MODEL_DEPENDENCY.get(k, "") for k in missing))) + "`"
    return text


def group_detail_markdown(label: str) -> str:
    """Grup açıklamasını Gradio Markdown olarak üretir (hedef kitle: ML bilmeyenler)."""
    group = GROUP_BY_LABEL[label]
    return f"### {group['label']}\n{group['desc']}\n\n{group_status_text(group)}"


# ---------------------------------------------------------------------------
# Temizlik ve benzersiz eğitim dizini
# ---------------------------------------------------------------------------
def cleanup_old_runs(max_age_seconds=24 * 3600):
    """Eski eğitim klasörlerini temizleyerek disk şişmesini önler."""
    try:
        os.makedirs(MODELS_ROOT, exist_ok=True)
        now = time.time()
        for name in os.listdir(MODELS_ROOT):
            path = os.path.join(MODELS_ROOT, name)
            if name.startswith("ag_models_") and os.path.isdir(path):
                if now - os.path.getmtime(path) > max_age_seconds:
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info("Eski eğitim klasörü temizlendi: %s", path)
    except Exception as exc:  # temizlik hataları uygulamayı durdurmasın
        logger.warning("Temizlik yapılamadı: %s", exc)


def create_unique_path():
    """Her eğitim için benzersiz bir model dizini oluşturur (çakışma yaşanmaz)."""
    os.makedirs(MODELS_ROOT, exist_ok=True)
    token = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    return os.path.join(MODELS_ROOT, f"ag_models_{token}")


# ---------------------------------------------------------------------------
# Adım 1: Veri Yükleme
# ---------------------------------------------------------------------------
def load_data(file) -> tuple:
    """
    CSV'yi okur, önizleme tablosu ve hedef değişken adaylarını döndürür.
    Yeni veri yüklenince önceki leaderboard sıfırlanır.
    """
    empty_leaderboard = pd.DataFrame(
        columns=["model", "score_val", "eval_metric", "pred_time_val", "fit_time"]
    )

    if file is None:
        return (
            pd.DataFrame({"Uyarı": ["Lütfen bir CSV dosyası yükleyin."]}),
            gr.Dropdown(choices=[], value=None),
            empty_leaderboard,
        )

    try:
        # Gradio gr.File, sürüme göre dosya nesnesi veya dosya yolu (str) döndürür.
        file_path = getattr(file, "name", file)
        df = pd.read_csv(file_path)
        logger.info("Veri yüklendi: %s satır, %s sütun", *df.shape)
        return df.head(5), gr.Dropdown(choices=list(df.columns), value=df.columns[0]), empty_leaderboard
    except Exception as exc:
        logger.error("CSV okunamadı:\n%s", traceback.format_exc())
        return (
            pd.DataFrame({"Hata": [str(exc)]}),
            gr.Dropdown(choices=[], value=None),
            empty_leaderboard,
        )


# ---------------------------------------------------------------------------
# Adım 3: Eğitim
# ---------------------------------------------------------------------------
def train_model(df, target_column, strategy, time_limit=0) -> tuple:
    """
    TabularPredictor.fit() çalıştırır ve leaderboard + metrik özeti döndürür.
    Her eğitimde benzersiz bir path kullanılır; eksik veriler AutoGluon
    tarafından varsayılan olarak işlenir, ekstra impute gerekmez.
    """
    status = {"info": "", "error": ""}

    # --- Doğrulamalar ---
    if df is None or df.empty:
        return _empty_result("Önce Adım 1'de veri yüklemeniz gerekiyor.")
    if not target_column or target_column not in df.columns:
        return _empty_result("Geçerli bir hedef değişken seçilmedi.")
    if strategy not in GROUP_BY_LABEL:
        return _empty_result("Geçersiz model stratejisi.")

    group = GROUP_BY_LABEL[strategy]
    hyperparameters, presets = group_hyperparameters(group)
    time_limit = float(time_limit) if (time_limit or 0) > 0 else None

    # Eski eğitim artıklarını temizle, benzersiz dizin oluştur
    cleanup_old_runs()
    model_path = create_unique_path()
    logger.info("Eğitim başlıyor. Grup=%s, path=%s", strategy, model_path)

    try:
        predictor = TabularPredictor(
            label=target_column,
            problem_type=None,   # AutoGluon problemi otomatik tespit etsin
            eval_metric=None,    # metriği de otomatik belirlesin
            path=model_path,
        )

        predictor.fit(
            train_data=df,
            presets=presets,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
            raise_on_no_models_fitted=False,  # bazı modeller başarısız olursa çökme
        )

        # --- Leaderboard + metrik özeti ---
        leaderboard = predictor.leaderboard(extra_info=False)
        problem_type = predictor.problem_type

        if not predictor.model_names:
            # Hiçbir model eğitilemediyse kullanıcıyı bilgilendir (ör. eksik bağımlılık)
            status["error"] = (
                "Hiçbir model eğitilemedi. Gerekli kütüphanelerin kurulu olduğundan emin olun "
                "(örn. `pip install autogluon.tabular[lightgbm,catboost,xgboost,fastai,torch]`)."
            )
            return leaderboard, status

        eval_metric = leaderboard["eval_metric"].iloc[0] if "eval_metric" in leaderboard else "N/A"
        best_row = leaderboard.iloc[0]  # leaderboard ilk satır en iyi modeldir
        status["info"] = (
            f"**{group['label']}** grubuyla eğitim tamamlandı.  \n"
            f"Problem Tipi: `{problem_type}` | Metrik: `{eval_metric}`  \n"
            f"En İyi Model: `{best_row['model']}` | Doğrulama Skoru: `{best_row['score_val']:.4f}`"
        )

        return leaderboard, status

    except Exception as exc:
        logger.error("Eğitim sırasında hata oluştu:\n%s", traceback.format_exc())
        status["error"] = f"Eğitim sırasında hata oluştu: {exc}"
        return _empty_result(status["error"])


def _empty_result(message: str) -> tuple:
    """Hata durumlarında ortak dönüş formatı."""
    empty_leaderboard = pd.DataFrame(
        columns=["model", "score_val", "eval_metric", "pred_time_val", "fit_time"]
    )
    return empty_leaderboard, {"info": "", "error": message}


# ---------------------------------------------------------------------------
# Gradio Arayüzü
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="AutoGluon ML Web Uygulaması") as demo:
        gr.Markdown(
            """
            # AutoGluon ile Uçtan Uca Makine Öğrenmesi
            1. **Veri Yükle** (CSV) → 2. **Açıklamayı oku, grubu seç** → 3. **Eğit** → Leaderboard'u gör.
            """
        )

        # Adım 1: Veri Yükleme
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Adım 1: Veri Yükleme")
                file_input = gr.File(label="CSV Dosyası", file_types=[".csv"])
                target_column = gr.Dropdown(label="Hedef Değişken (Target Column)", choices=[], value=None)
            with gr.Column(scale=2):
                data_preview = gr.Dataframe(label="Veri Önizleme (ilk 5 satır)", interactive=False)

        # Adım 2: Model Grubu Seçimi
        gr.Markdown("### Adım 2: Model Grubunu Seç")
        with gr.Row():
            with gr.Column(scale=1):
                strategy = gr.Dropdown(label="Model Grubu", choices=STRATEGY_LABELS, value=STRATEGY_LABELS[0])
            with gr.Column(scale=2):
                strategy_desc = gr.Markdown(group_detail_markdown(STRATEGY_LABELS[0]))
        time_limit = gr.Number(label="Zaman Limiti (saniye, 0 = limitsiz)", value=0, precision=0, minimum=0)

        # Adım 3: Eğitim ve Derecelendirme
        gr.Markdown("### Adım 3: Eğitim ve Derecelendirme")
        train_button = gr.Button("Modeli Eğit", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                leaderboard = gr.Dataframe(label="AutoGluon Leaderboard", interactive=False)
            with gr.Column(scale=1):
                result_summary = gr.Markdown("Sonuçlar burada görünecek.")

        # --- Olay bağlantıları (event handlers) ---
        file_input.change(
            fn=load_data,
            inputs=file_input,
            outputs=[data_preview, target_column, leaderboard],
        )
        strategy.select(
            fn=group_detail_markdown,
            inputs=strategy,
            outputs=strategy_desc,
        )
        train_button.click(
            fn=_wrap_train,
            inputs=[data_preview, target_column, strategy, time_limit],
            outputs=[leaderboard, result_summary],
        )

    return demo


def format_summary_markdown(status: dict) -> str:
    """Sonuç özetini Gradio Markdown'a uygun hale getirir."""
    if status.get("error"):
        return f"## ❌ {status['error']}"
    if status.get("info"):
        return f"## ✅ Eğitim Tamamlandı\n{status['info']}"
    return "Sonuçlar burada görünecek."


def _wrap_train(df, target_column, strategy, time_limit=0):
    """train_model çıktısını (leaderboard, durum) Markdown özetine dönüştürür."""
    leaderboard, status = train_model(df, target_column, strategy, time_limit)
    return leaderboard, format_summary_markdown(status)


if __name__ == "__main__":
    cleanup_old_runs()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
