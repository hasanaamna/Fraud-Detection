
import { useCallback, useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  CreditCard,
  FileSearch,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  RefreshCw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  X,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

type Stats = {
  model: string;
  total_transactions: number;
  actual_frauds: number;
  predicted_frauds: number;
  fraud_rate: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  pr_auc: number;
  threshold: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
};

type TrendPoint = {
  hour: number;
  transactions: number;
  frauds: number;
  fraud_rate: number;
};

type TestTransaction = {
  transaction: Record<string, number>;
  time: number;
  amount: number;
  actual_class: number;
  predicted_class: number;
  fraud_probability: number;
  decision_threshold: number;
};

type FeatureContribution = {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "toward_fraud" | "away_from_fraud" | "neutral";
};

type Explanation = {
  model: string;
  prediction: number;
  label: string;
  fraud_probability: number;
  threshold: number;
  base_value: number;
  reconstructed_probability: number;
  explanation_consistent: boolean;
  top_features: FeatureContribution[];
  positive_contributors: FeatureContribution[];
  negative_contributors: FeatureContribution[];
  feature_note: string;
};

type Prediction = {
  prediction: number;
  label: string;
  fraud_probability: number;
  fraud_probability_percent: number;
  risk_level: string;
  threshold: number;
  model: string;
  message: string;
};

const featureNames = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

const initialTransaction: Record<string, number> = {
  Time: 0,
  V1: -1.359807, V2: -0.072781, V3: 2.536347,
  V4: 1.378155, V5: -0.338321, V6: 0.462388,
  V7: 0.239599, V8: 0.098698, V9: 0.363787,
  V10: 0.090794, V11: -0.5516, V12: -0.617801,
  V13: -0.99139, V14: -0.311169, V15: 1.468177,
  V16: -0.470401, V17: 0.207971, V18: 0.025791,
  V19: 0.403993, V20: 0.251412, V21: -0.018307,
  V22: 0.277838, V23: -0.110474, V24: 0.066928,
  V25: 0.128539, V26: -0.189115, V27: 0.133558,
  V28: -0.021053, Amount: 149.62,
};

const number = (value: number) => value.toLocaleString("en-US");

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

const money = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [apiStatus, setApiStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");

  const [stats, setStats] = useState<Stats | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [samples, setSamples] = useState<TestTransaction[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const [transaction, setTransaction] = useState(initialTransaction);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explanationError, setExplanationError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dashboardError, setDashboardError] = useState("");

  const loadDashboard = useCallback(async () => {
    setApiStatus("checking");
    setDashboardError("");

    try {
      const [
        healthResponse,
        statsResponse,
        trendResponse,
        legitimateSamplesResponse,
        fraudSamplesResponse,
      ] = await Promise.all([
        fetch(`${API_URL}/health`),
        fetch(`${API_URL}/dashboard/stats`),
        fetch(`${API_URL}/dashboard/fraud-trend`),
        // Fetch both classes explicitly. The first rows in the test CSV are
        // mostly legitimate, so a single unfiltered request can hide fraud cases.
        fetch(`${API_URL}/dashboard/sample-transactions?limit=50&actual_class=0`),
        fetch(`${API_URL}/dashboard/sample-transactions?limit=50&actual_class=1`),
      ]);

      if (
        !healthResponse.ok ||
        !statsResponse.ok ||
        !trendResponse.ok ||
        !legitimateSamplesResponse.ok ||
        !fraudSamplesResponse.ok
      ) {
        throw new Error(
          "The API could not load all dashboard data. Check the backend and try again.",
        );
      }

      const [statsData, trendData, legitimateData, fraudData] = await Promise.all([
        statsResponse.json(),
        trendResponse.json(),
        legitimateSamplesResponse.json(),
        fraudSamplesResponse.json(),
      ]);

      // Put legitimate and fraud examples into one selector and label each
      // with its saved probability, so users can test both kinds of records.
      const combinedSamples = [
        ...(legitimateData.items as TestTransaction[]),
        ...(fraudData.items as TestTransaction[]),
      ];

      setStats(statsData as Stats);
      setTrend(trendData.points as TrendPoint[]);
      setSamples(combinedSamples);
      setSelectedIndex(0);
      setTransaction(
        combinedSamples[0] ? { ...combinedSamples[0].transaction } : { ...initialTransaction },
      );
      setPrediction(null);
      setExplanation(null);
      setExplanationError("");
      setError("");
      setApiStatus("online");
    } catch (err) {
      setApiStatus("offline");
      setDashboardError(
        err instanceof Error ? err.message : "Unable to load dashboard data.",
      );
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  const selectedSample = samples[selectedIndex] ?? null;

  function chooseSample(index: number) {
    const sample = samples[index];
    if (!sample) return;

    setSelectedIndex(index);
    setTransaction({ ...sample.transaction });
    setPrediction(null);
    setExplanation(null);
    setExplanationError("");
    setError("");
  }

  function updateTransaction(key: string, value: string) {
    setTransaction((current) => ({
      ...current,
      [key]: value === "" ? 0 : Number(value),
    }));
    setPrediction(null);
    setExplanation(null);
    setExplanationError("");
  }

  async function runPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setExplanationError("");
    setPrediction(null);
    setExplanation(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transaction }),
      });

      const result = await response.json();
      if (!response.ok) {
        const detail = result.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || "Prediction request failed.",
        );
      }

      setPrediction(result as Prediction);

      // Explanation is a separate request so a SHAP failure does not hide
      // an otherwise valid prediction.
      try {
        const explainResponse = await fetch(`${API_URL}/explain`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transaction }),
        });
        const explainResult = await explainResponse.json();
        if (!explainResponse.ok) {
          const detail = explainResult.detail;
          throw new Error(
            typeof detail === "string"
              ? detail
              : detail?.message || "Could not load feature explanations.",
          );
        }
        setExplanation(explainResult as Explanation);
      } catch (err) {
        setExplanationError(
          err instanceof Error
            ? err.message
            : "The prediction succeeded, but the explanation could not be loaded.",
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to reach the prediction service.",
      );
    } finally {
      setLoading(false);
    }
  }

  function resetTransaction() {
    setTransaction({ ...initialTransaction });
    setSelectedIndex(-1);
    setPrediction(null);
    setExplanation(null);
    setExplanationError("");
    setError("");
  }

  const navItems = [
    { name: "Overview", icon: LayoutDashboard },
    { name: "Transactions", icon: CreditCard },
    { name: "Fraud Analysis", icon: FileSearch },
  ];

  const confusionData = stats
    ? [
        { name: "True negatives", value: stats.true_negatives },
        { name: "False positives", value: stats.false_positives },
        { name: "False negatives", value: stats.false_negatives },
        { name: "True positives", value: stats.true_positives },
      ]
    : [];

  const confusionColors = ["#27D6B1", "#F3B64B", "#F36F83", "#8A7CF7"];

  const sidebar = (
    <>
      <div className="brand">
        <div className="brand-icon"><Shield size={23} /></div>
        <div>
          <div className="brand-name">Sentinel<span>AI</span></div>
          <div className="brand-caption">FRAUD INTELLIGENCE</div>
        </div>
        <button
          className="icon-button mobile-close"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
        >
          <X size={19} />
        </button>
      </div>

      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list">
        {navItems.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.name}
              className={`nav-item ${activePage === item.name ? "active" : ""}`}
              onClick={() => {
                setActivePage(item.name);
                setSidebarOpen(false);

                const target =
                  item.name === "Transactions"
                    ? "risk-checker"
                    : item.name === "Fraud Analysis"
                      ? "analysis"
                      : "top";

                document.getElementById(target)?.scrollIntoView({
                  behavior: "smooth",
                });
              }}
            >
              <Icon size={18} />
              <span>{item.name}</span>
              {item.name === "Transactions" && (
                <span className="nav-count">{number(samples.length)}</span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-spacer" />

      <div className="sidebar-security">
        <div className="security-icon"><ShieldCheck size={18} /></div>
        <div>
          <strong>Model protection</strong>
          <p>Local inference enabled</p>
        </div>
        <span className="live-dot" />
      </div>

      <div className="sidebar-footer">
        <span className="avatar">AI</span>
        <div className="profile-text">
          <strong>Fraud Analyst</strong>
          <span>Local workspace</span>
        </div>
        <ChevronDown size={16} className="muted-icon" />
      </div>
    </>
  );

  return (
    <div className="app-shell" id="top">
      {sidebarOpen && (
        <button
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        {sidebar}
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button menu-toggle"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={21} />
            </button>

            <div className="breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb-slash">/</span>
              <strong>{activePage}</strong>
            </div>
          </div>

          <div className="topbar-right">
            <div className={`connection-pill ${apiStatus}`}>
              <span className="live-dot" />
              {apiStatus === "online"
                ? "API Connected"
                : apiStatus === "offline"
                  ? "API Offline"
                  : "Connecting"}
            </div>

            <button
              className="icon-button refresh-button"
              onClick={() => void loadDashboard()}
              title="Refresh dashboard data"
              aria-label="Refresh dashboard data"
            >
              <RefreshCw size={17} />
            </button>

            <div className="topbar-avatar">AA</div>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-section">
            <div>
              <div className="eyebrow">
                <Sparkles size={14} /> FRAUD INTELLIGENCE PLATFORM
              </div>
              <h1>Fraud analytics <span>overview</span></h1>
              <p className="welcome-subtitle">
                Monitor model performance and evaluate transaction risk.
              </p>
            </div>

            <div className="date-chip">
              <Activity size={16} />
              <span>Model evaluation</span>
              <span className="date-separator" />
              <span>Held-out test set</span>
            </div>
          </section>

          {dashboardError && (
            <div className="error-result" role="alert">
              <AlertTriangle size={18} />
              <div>
                <strong>Dashboard data unavailable</strong>
                <p>{dashboardError}</p>
                <button
                  className="button-secondary"
                  onClick={() => void loadDashboard()}
                >
                  <RefreshCw size={15} /> Retry
                </button>
              </div>
            </div>
          )}

          <section className="metric-grid">
            <MetricCard
              icon={<CreditCard size={19} />}
              label="TEST TRANSACTIONS"
              value={stats ? number(stats.total_transactions) : "—"}
              note="Held-out evaluation set"
              tone="blue"
            />
            <MetricCard
              icon={<ShieldAlert size={19} />}
              label="FRAUD DETECTED"
              value={stats ? number(stats.true_positives) : "—"}
              note={stats ? `Of ${number(stats.actual_frauds)} actual fraud cases` : "Loading actual results"}
              tone="purple"
            />
            <MetricCard
              icon={<CircleDollarSign size={19} />}
              label="PRECISION"
              value={stats ? percent(stats.precision) : "—"}
              note="Flagged transactions that were fraud"
              tone="teal"
            />
            <MetricCard
              icon={<TrendingUp size={19} />}
              label="RECALL"
              value={stats ? percent(stats.recall) : "—"}
              note="Fraud cases successfully detected"
              tone="orange"
            />
          </section>

          <section className="analysis-grid" id="analysis">
            <div className="panel trend-panel">
              <div className="panel-header">
                <div>
                  <h2>Fraud rate trend</h2>
                  <p>Actual fraud rate · held-out test dataset</p>
                </div>
                <span className="tag tag-success">
                  <Activity size={13} /> Live data
                </span>
              </div>

              <div className="chart-area">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={trend}
                    margin={{ top: 12, right: 10, left: -18, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="fraudGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#27D6B1" stopOpacity={0.27} />
                        <stop offset="100%" stopColor="#27D6B1" stopOpacity={0} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      stroke="#233142"
                      strokeDasharray="4 5"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="hour"
                      tick={{ fill: "#8492A6", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(hour) => `${hour}h`}
                    />

                    <YAxis
                      tick={{ fill: "#8492A6", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#111C2B",
                        border: "1px solid #2B3B4F",
                        borderRadius: 10,
                        color: "#E8EEF7",
                      }}
                      formatter={(value, name) => [
                        name === "fraud_rate"
                          ? percent(Number(value))
                          : number(Number(value)),
                        name === "fraud_rate"
                          ? "Fraud rate"
                          : name === "frauds"
                            ? "Actual frauds"
                            : "Transactions",
                      ]}
                      labelFormatter={(hour) => `Elapsed hour ${hour}`}
                    />

                    <Area
                      type="monotone"
                      dataKey="fraud_rate"
                      stroke="#27D6B1"
                      strokeWidth={2.5}
                      fill="url(#fraudGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-footer">
                <span><span className="legend-dot teal-dot" /> Actual test-set fraud rate</span>
                <span>{trend.length} time windows</span>
              </div>
            </div>

            <div className="panel model-panel">
              <div className="panel-header">
                <div>
                  <h2>Model performance</h2>
                  <p>{stats?.model ?? "Loading model"} · held-out test results</p>
                </div>
                <span className="tag tag-success">Evaluated</span>
              </div>

              <div className="model-score">
                <div className="score-ring">
                  <div className="score-ring-inner">
                    <strong>{stats ? percent(stats.f1_score) : "—"}</strong>
                    <span>F1 SCORE</span>
                  </div>
                </div>
                <div className="score-description">
                  <strong>Balanced detection</strong>
                  <p>Precision and recall combined into one score.</p>
                </div>
              </div>

              <div className="performance-list">
                <PerformanceRow label="ROC-AUC" value={stats?.roc_auc} />
                <PerformanceRow label="PR-AUC" value={stats?.pr_auc} />
                <PerformanceRow label="Precision" value={stats?.precision} />
                <PerformanceRow label="Recall" value={stats?.recall} />
              </div>

              <div className="model-footer">
                <span><span className="live-dot" /> {stats?.model ?? "Model pending"}</span>
                <span>
                  Threshold: {stats ? stats.threshold.toFixed(6) : "—"}
                </span>
              </div>
            </div>
          </section>

          <section className="bottom-grid">
            <div className="panel confusion-panel">
              <div className="panel-header">
                <div>
                  <h2>Prediction breakdown</h2>
                  <p>Actual outcomes on the held-out test set</p>
                </div>
              </div>

              <div className="confusion-content">
                <div className="donut-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={confusionData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius="64%"
                        outerRadius="88%"
                        paddingAngle={2}
                        stroke="none"
                      >
                        {confusionData.map((entry, index) => (
                          <Cell
                            key={entry.name}
                            fill={confusionColors[index]}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: "#111C2B",
                          border: "1px solid #2B3B4F",
                          borderRadius: 10,
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>

                  <div className="donut-center">
                    <strong>{stats ? number(stats.total_transactions) : "—"}</strong>
                    <span>TEST ROWS</span>
                  </div>
                </div>

                <div className="confusion-legend">
                  {confusionData.map((item, index) => (
                    <div className="confusion-item" key={item.name}>
                      <span
                        className="legend-square"
                        style={{ background: confusionColors[index] }}
                      />
                      <div>
                        <span>{item.name}</span>
                        <strong>{number(item.value)}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="panel insight-panel">
              <div className="panel-header">
                <div>
                  <h2>Model insights</h2>
                  <p>What the current evaluation tells us</p>
                </div>
                <Sparkles size={18} className="insight-spark" />
              </div>

              <div className="insight-item">
                <div className="insight-icon insight-green"><CheckCircle2 size={18} /></div>
                <div>
                  <strong>{stats ? `${number(stats.true_positives)} fraud cases detected` : "Loading results"}</strong>
                  <p>
                    {stats
                      ? `Detected ${number(stats.true_positives)} of ${number(stats.actual_frauds)} actual fraudulent test transactions.`
                      : "Retrieving evaluation metrics."}
                  </p>
                </div>
              </div>

              <div className="insight-item">
                <div className="insight-icon insight-yellow"><AlertTriangle size={18} /></div>
                <div>
                  <strong>{stats ? `${number(stats.false_negatives)} missed fraud cases` : "Loading results"}</strong>
                  <p>These are fraudulent test rows that the model classified as legitimate.</p>
                </div>
              </div>

              <div className="insight-item">
                <div className="insight-icon insight-blue"><Clock3 size={18} /></div>
                <div>
                  <strong>Research dataset</strong>
                  <p>These are historical benchmark results, not live banking traffic.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="panel risk-panel" id="risk-checker">
            <div className="risk-heading">
              <div>
                <div className="eyebrow">
                  <ShieldCheck size={14} /> LIVE API INFERENCE
                </div>
                <h2>Transaction risk analyzer</h2>
                <p>
                  Test a real held-out transaction or submit the example features
                  to your locally hosted model.
                </p>
              </div>

              <div className={`connection-pill ${apiStatus}`}>
                <span className="live-dot" />
                {apiStatus === "online"
                  ? "Ready for prediction"
                  : apiStatus === "offline"
                    ? "Backend unavailable"
                    : "Checking backend"}
              </div>
            </div>

            <div className="sample-selector">
              <label className="field">
                <span>Held-out test transaction</span>
                <select
                  value={selectedIndex}
                  onChange={(event) => chooseSample(Number(event.target.value))}
                  disabled={!samples.length}
                >
                  <option value={-1}>Manual example transaction</option>
                  {samples.map((sample, index) => (
                    <option key={index} value={index}>
                      {sample.actual_class === 1 ? "FRAUD" : "Legitimate"} ·
                      #{index + 1} · {money(sample.amount)} · Saved risk:{" "}
                      {percent(sample.fraud_probability)}
                    </option>
                  ))}
                </select>
              </label>

              {selectedSample && (
                <div className="sample-note">
                  <Activity size={15} />
                  <span>
                    Actual label: <strong>{selectedSample.actual_class === 1 ? "Fraud" : "Legitimate"}</strong>
                    {" · "}
                    Saved prediction: <strong>{selectedSample.predicted_class === 1 ? "Fraud" : "Legitimate"}</strong>
                    {" · "}
                    Saved probability: <strong>{percent(selectedSample.fraud_probability)}</strong>
                  </span>
                </div>
              )}
            </div>

            <form onSubmit={runPrediction}>
              <div className="form-grid">
                <label className="field">
                  <span>Transaction amount ($)</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    required
                    value={transaction.Amount}
                    onChange={(event) => updateTransaction("Amount", event.target.value)}
                  />
                </label>

                <label className="field">
                  <span>Elapsed time (seconds)</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    required
                    value={transaction.Time}
                    onChange={(event) => updateTransaction("Time", event.target.value)}
                  />
                </label>
              </div>

              <div className="sample-note">
                <AlertTriangle size={15} />
                <span>
                  V1–V28 are anonymized PCA features. For meaningful testing,
                  select an actual held-out transaction and keep its features
                  together. Editing amount or time changes that sample.
                </span>
              </div>

              <div className="form-actions">
                <button
                  type="button"
                  className="button-secondary"
                  onClick={resetTransaction}
                >
                  <RefreshCw size={16} /> Reset sample
                </button>

                <button
                  type="submit"
                  className="button-primary"
                  disabled={loading || apiStatus !== "online"}
                >
                  {loading
                    ? <LoaderCircle size={17} className="spin" />
                    : <Shield size={17} />}
                  {loading ? "Analyzing..." : "Analyze transaction"}
                  {!loading && <ArrowUpRight size={16} />}
                </button>
              </div>
            </form>

            {error && (
              <div className="error-result" role="alert">
                <AlertTriangle size={18} />
                <div>
                  <strong>Prediction failed</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {prediction && (
              <div className={`prediction-result risk-${prediction.risk_level.toLowerCase()}`}>
                <div className="result-icon">
                  {prediction.prediction === 1
                    ? <ShieldAlert size={24} />
                    : <ShieldCheck size={24} />}
                </div>

                <div className="result-main">
                  <span className="result-eyebrow">MODEL ASSESSMENT</span>
                  <h3>{prediction.label}</h3>
                  <p>{prediction.message}</p>

                  {selectedSample && (
                    <p className="actual-comparison">
                      Actual test label:{" "}
                      <strong>
                        {selectedSample.actual_class === 1 ? "Fraud" : "Legitimate"}
                      </strong>
                      {" · "}
                      {prediction.prediction === selectedSample.actual_class
                        ? "Prediction matches the known label"
                        : "Prediction differs from the known label"}
                    </p>
                  )}
                </div>

                <div className="result-probability">
                  <strong>{prediction.fraud_probability_percent.toFixed(2)}%</strong>
                  <span>Fraud probability</span>
                  <span className="result-threshold">
                    Threshold {(prediction.threshold * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}

            {prediction && (
              <section className="explanation-panel" aria-live="polite">
                <div className="explanation-header">
                  <div className="explanation-title-icon"><Sparkles size={19} /></div>
                  <div>
                    <span className="result-eyebrow">EXPLAINABLE AI</span>
                    <h3>Why did the model make this prediction?</h3>
                    <p>SHAP shows how each feature moved this prediction relative to the model baseline.</p>
                  </div>
                  {explanation && (
                    <span className={`explanation-status ${explanation.explanation_consistent ? "is-consistent" : "is-warning"}`}>
                      {explanation.explanation_consistent ? "Explanation verified" : "Check explanation scale"}
                    </span>
                  )}
                </div>

                {explanationError && (
                  <div className="explanation-error" role="status">
                    <AlertTriangle size={16} />
                    <span>Prediction succeeded, but SHAP details could not be loaded: {explanationError}</span>
                  </div>
                )}

                {!explanation && !explanationError && (
                  <div className="explanation-loading"><LoaderCircle size={17} className="spin" /> Calculating feature contributions…</div>
                )}

                {explanation && (
                  <>
                    <div className="explanation-summary">
                      <div>
                        <span>Fraud probability</span>
                        <strong>{percent(explanation.fraud_probability)}</strong>
                      </div>
                      <div>
                        <span>Model baseline</span>
                        <strong>{percent(explanation.base_value)}</strong>
                      </div>
                      <div>
                        <span>Features analyzed</span>
                        <strong>{number(30)}</strong>
                      </div>
                    </div>

                    <div className="contribution-heading">
                      <h4>Top feature contributions</h4>
                      <span>Sorted by absolute SHAP impact</span>
                    </div>
                    <div className="contribution-list">
                      {explanation.top_features.map((item) => {
                        const maxImpact = Math.max(
                          ...explanation.top_features.map((feature) => Math.abs(feature.shap_value)),
                          0.000001,
                        );
                        const width = Math.max(2, (Math.abs(item.shap_value) / maxImpact) * 100);
                        const towardFraud = item.shap_value > 0;
                        return (
                          <div className="contribution-row" key={item.feature}>
                            <div className="contribution-meta">
                              <strong>{item.feature}</strong>
                              <span>{towardFraud ? "Toward fraud" : item.shap_value < 0 ? "Away from fraud" : "Neutral"}</span>
                              <b className={towardFraud ? "contribution-positive" : "contribution-negative"}>
                                {item.shap_value > 0 ? "+" : ""}{item.shap_value.toFixed(5)}
                              </b>
                            </div>
                            <div className="contribution-track">
                              <div
                                className={`contribution-fill ${towardFraud ? "positive" : "negative"}`}
                                style={{ width: `${width}%` }}
                              />
                            </div>
                            <div className="contribution-value">Feature value: {item.feature_value.toLocaleString("en-US", { maximumFractionDigits: 5 })}</div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="explanation-disclaimer">
                      <AlertTriangle size={16} />
                      <p>{explanation.feature_note} Positive SHAP values push toward fraud; negative values push away from fraud. Contributions are not individual fraud probabilities or proof of causation.</p>
                    </div>
                    {!explanation.explanation_consistent && (
                      <div className="explanation-error" role="alert">
                        SHAP contributions did not reconstruct the displayed probability within tolerance. Treat this explanation as provisional until the output scale is checked.
                      </div>
                    )}
                  </>
                )}
              </section>
            )}
          </section>

          <footer className="page-footer">
            <div><Shield size={15} /> SentinelAI <span>·</span> Fraud Intelligence</div>
            <span>Research prototype · Local model inference</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  note,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  note: string;
  tone: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div className={`metric-icon ${tone}`}>{icon}</div>
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

function PerformanceRow({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  const validValue = value ?? 0;

  return (
    <div className="performance-row">
      <div className="performance-label">
        <span>{label}</span>
        <strong>{value === undefined ? "—" : percent(validValue)}</strong>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.max(0, Math.min(1, validValue)) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default App;
