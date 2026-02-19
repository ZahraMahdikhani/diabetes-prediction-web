// app/page.tsx
"use client";

import { useState } from "react";
import DiabetesForm from "@/components/diabetes-form";
import ResultsDisplay from "@/components/results-display";

export default function Home() {
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (formData: Record<string, any>) => {
    setIsLoading(true);

    try {
      // محاسبه BMI در کلاینت (برای نمایش فوری و اطمینان)
      const height = parseFloat(formData.height_cm);
      const weight = parseFloat(formData.weight_kg);

      if (height > 0 && weight > 0) {
        const heightInMeters = height / 100;
        formData.BMI = Number((weight / (heightInMeters * heightInMeters)).toFixed(1));
      }

      // حذف فیلدهای موقتی قبل از ارسال به API
      delete formData.height_cm;
      delete formData.weight_kg;

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`خطا ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("خطا در ارتباط با سرور:", error);
      alert("مشکلی پیش آمد. لطفاً دوباره تلاش کنید.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-50 via-blue-50 to-white py-8 px-4 md:px-8">
      <div className="max-w-5xl mx-auto">
        {results ? (
          <ResultsDisplay results={results} onReset={() => setResults(null)} />
        ) : (
          <DiabetesForm onSubmit={handleSubmit} isLoading={isLoading} />
        )}
      </div>
    </main>
  );
}