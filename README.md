# 🎬 Movie Review Sentiment Intelligence — BERT + Streamlit

نظام تحليل مشاعر لمراجعات الأفلام مبني على **BERT** متدرب (fine-tuned) على داتاست **Stanford IMDB** (50,000 مراجعة)، مع ميزة إضافية بتكسر النظام العادي: تحليل المشاعر على مستوى الـ **aspects** (التمثيل، القصة، الموسيقى، الصورة، الإخراج) — يعني تقدر تعرف إن المراجعة "إيجابية عمومًا بس سلبية بالنسبة للقصة" مثلًا.

## ✨ المميزات

- تصنيف Positive / Negative مع نسبة ثقة (confidence)
- Aspect-based breakdown: تمثيل، قصة، موسيقى، صورة/مؤثرات، إخراج
- تحليل مراجعة واحدة أو ملف كامل (batch analysis)
- واجهة Streamlit جاهزة للعرض على شركات/عملاء

## 🗂️ محتويات الريبو

```
.
├── app.py              # واجهة Streamlit
├── push_to_hub.py       # سكريبت لرفع الموديل على Hugging Face Hub (تشغّله مرة واحدة)
├── requirements.txt
├── notebook/            # النوتبوك الأصلي (تدريب + تحليل + رسومات)
└── README.md
```

## 🚀 خطوات النشر (مرة واحدة بس)

الموديل حجمه حوالي 400+ ميجا، فمينفعش يترفع مباشرة على GitHub. الحل الأنضف إنك ترفعيه على **Hugging Face Hub** (مجاني) والـ Streamlit app تحمّله من هناك تلقائيًا.

### 1) ارفعي الموديل على Hugging Face Hub

```bash
pip install huggingface_hub transformers
huggingface-cli login
```

عدّلي `REPO_ID` في `push_to_hub.py` لاسم المستخدم بتاعك، وبعدين:

```bash
python push_to_hub.py
```

### 2) عدّلي app.py

في أول الملف، غيّري السطر ده:

```python
MODEL_REPO = "your-username/bert-imdb-sentiment"
```

لاسم الريبو اللي طلع عندك من الخطوة اللي فاتت.

### 3) ارفعي المشروع على GitHub

```bash
git init
git add .
git commit -m "Initial commit: BERT sentiment analysis app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### 4) انشري على Streamlit Community Cloud (مجاني)

1. روحي على [share.streamlit.io](https://share.streamlit.io)
2. اربطي حساب GitHub بتاعك
3. اختاري الريبو → main branch → `app.py` كـ entry point
4. Deploy

بعد كام دقيقة هيبقى عندك لينك عام تقدري تبعتيه لأي شركة تجرب الديمو مباشرة من المتصفح.

## 📊 أداء الموديل (Test Set)

| Metric | Value |
|---|---|
| Accuracy | ~93% |
| Precision | ~93% |
| Recall | ~93% |
| F1-Score | ~93% |

> عدّلي الأرقام دي في `app.py` (`REPORTED_METRICS`) بالأرقام الحقيقية اللي طلعت معاكي من الـ evaluation section في النوتبوك.

## 🧠 التفاصيل التقنية

- **Base model:** `bert-base-uncased`
- **Dataset:** Stanford IMDB Large Movie Review Dataset (Hugging Face: `stanfordnlp/imdb`)
- **Max sequence length:** 256 tokens
- **Fine-tuning:** 3 epochs, learning rate 2e-5, batch size 16
- **Aspect detection:** keyword-based clause splitting (مش موديل منفصل — بيستخدم نفس الموديل الأساسي على الجمل اللي بتتكلم عن كل aspect)

## ⚠️ حدود النظام

- تحليل الـ aspects مبني على keywords، فلو المراجعة اتكلمت عن حاجة من غير الكلمات المتوقعة مش هيتلقطها
- الموديل اتدرب على مراجعات إنجليزي بس
