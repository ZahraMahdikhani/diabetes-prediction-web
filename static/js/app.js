// app.js - نسخه دو مرحله‌ای با اعتبارسنجی بهتر

const form = document.getElementById("diabetesForm");
const sections = document.querySelectorAll(".form-section");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const submitBtn = document.getElementById("submitBtn");
const progressBar = document.getElementById("progressBar"); // اگر progress bar نداری، این خط رو کامنت کن

let currentSection = 0;

// شروع
function init() {
    showSection(0);
    updateButtons();
    // اگر progress bar داری:
    if (progressBar) updateProgressBar();
}

function showSection(n) {
    sections.forEach(s => s.classList.add("hidden"));
    sections[n].classList.remove("hidden");
    currentSection = n;

    // اسکرول نرم به ابتدای بخش جدید
    sections[n].scrollIntoView({ behavior: "smooth", block: "start" });

    updateButtons();
    if (progressBar) updateProgressBar();
}

function updateProgressBar() {
    if (!progressBar) return;
    const progress = ((currentSection + 1) / sections.length) * 100;
    progressBar.style.width = progress + "%";
}

function updateButtons() {
    if (prevBtn) prevBtn.style.display = currentSection > 0 ? "inline-flex" : "none";
    if (nextBtn) nextBtn.style.display = currentSection < sections.length - 1 ? "inline-flex" : "none";
    if (submitBtn) submitBtn.style.display = currentSection === sections.length - 1 ? "inline-flex" : "none";
}

// اعتبارسنجی بخش فعلی
function validateCurrentSection() {
    const current = sections[currentSection];
    const requiredElements = current.querySelectorAll('[required]');
    let isValid = true;

    requiredElements.forEach(el => {
        if (el.type === 'radio') {
            // چک کردن گروه رادیو
            const groupName = el.name;
            const radios = current.querySelectorAll(`input[name="${groupName}"]`);
            const anyChecked = Array.from(radios).some(r => r.checked);
            if (!anyChecked) {
                isValid = false;
                // می‌تونی به parent کلاس error اضافه کنی
            }
        } else if (el.type === 'number' || el.tagName === 'SELECT') {
            if (!el.value.trim()) {
                isValid = false;
            }
        }
    });

    return isValid;
}

// دکمه بعدی
if (nextBtn) {
    nextBtn.addEventListener("click", () => {
        if (validateCurrentSection()) {
            currentSection++;
            showSection(currentSection);
        } else {
            alert("لطفاً تمام فیلدهای اجباری این بخش را پر کنید.");
        }
    });
}

// دکمه قبلی
if (prevBtn) {
    prevBtn.addEventListener("click", () => {
        currentSection--;
        if (currentSection < 0) currentSection = 0;
        showSection(currentSection);
    });
}

// قبل از submit نهایی چک کن
form.addEventListener("submit", (e) => {
    if (!validateCurrentSection()) {
        e.preventDefault();
        alert("لطفاً تمام موارد را تکمیل کنید.");
    }
});

// شروع برنامه
init();