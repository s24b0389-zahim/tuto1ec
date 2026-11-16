import random
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="JIE42903 - Evolutionary Computing Assignment", layout="wide"
)

st.title("📺 TV Program Scheduling Using Genetic Algorithm")
st.markdown(
    "**Student Name:** MUHAMMAD ZAHIN BIN SALIMI | **Matric No:** S24B0389"
)
st.markdown(
    "**Course:** JIE42903 – Evolutionary Computing | **Lecturer:** Ts. Dr. Fakhitah"
    " Ridzuan"
)
st.divider()

# 1. Load and display modified dataset
st.subheader("📁 Modified Program Ratings Dataset")
try:
  df_ratings = pd.read_csv("program_ratings.csv")
  st.dataframe(df_ratings, use_container_width=True)
except FileNotFoundError:
  st.error(
      "❌ 'program_ratings.csv' not found. Please upload it to your repository."
  )


# Genetic Algorithm Class
class GeneticScheduler:

  def __init__(self, programs, pop_size=50, generations=100):
    self.programs = programs
    self.pop_size = pop_size
    self.generations = generations

  def fitness(self, chromosome):
    total_rating = sum(
        self.programs.loc[self.programs["Program_ID"] == p][
            "Modified_Rating"
        ].values[0]
        for p in chromosome
    )
    return total_rating

  def run(self, co_r, mut_r):
    prog_ids = self.programs["Program_ID"].tolist()
    population = [random.sample(prog_ids, len(prog_ids)) for _ in range(self.pop_size)]

    best_schedule = None
    best_fitness = -float("inf")

    for _ in range(self.generations):
      scored_pop = [(ind, self.fitness(ind)) for ind in population]
      scored_pop.sort(key=lambda x: x[1], reverse=True)

      if scored_pop[0][1] > best_fitness:
        best_fitness = scored_pop[0][1]
        best_schedule = scored_pop[0][0]

      selected = [ind for ind, fit in scored_pop[: self.pop_size // 2]]
      next_generation = selected.copy()

      while len(next_generation) < self.pop_size:
        p1, p2 = random.sample(selected, 2)
        if random.random() < co_r:
          cut = len(p1) // 2
          child = p1[:cut] + [item for item in p2 if item not in p1[:cut]]
        else:
          child = p1.copy()

        if random.random() < mut_r:
          idx1, idx2 = random.sample(range(len(child)), 2)
          child[idx1], child[idx2] = child[idx2], child[idx1]

        next_generation.append(child)

      population = next_generation

    # Format schedule into table rows
    scheduled_rows = []
    time_slots = [
        "08:00 AM",
        "09:00 AM",
        "10:00 AM",
        "11:00 AM",
        "12:00 PM",
        "01:00 PM",
        "02:00 PM",
        "03:00 PM",
        "05:00 PM",
        "07:00 PM",
        "09:00 PM",
    ]

    for i, prog_id in enumerate(best_schedule[: len(time_slots)]):
      row = self.programs[self.programs["Program_ID"] == prog_id].iloc[0]
      scheduled_rows.append({
          "Time Slot": time_slots[i],
          "Program ID": row["Program_ID"],
          "Program Name": row["Program_Name"],
          "Genre": row["Genre"],
          "Duration (Hrs)": row["Duration_Hours"],
          "Rating": row["Modified_Rating"],
      })

    return pd.DataFrame(scheduled_rows), best_fitness


# Sidebar Parameter Controls (as required by assignment)
st.sidebar.header("⚙️ GA Parameters")
co_r = st.sidebar.slider(
    "CO_R (Crossover Rate)",
    min_value=0.0,
    max_value=0.95,
    value=0.8,
    step=0.05,
)
mut_r = st.sidebar.slider(
    "MUT_R (Mutation Rate)",
    min_value=0.01,
    max_value=0.05,
    value=0.02,
    step=0.01,
)

st.sidebar.divider()
run_trials_btn = st.sidebar.button("🚀 Run 3 Experimental Trials")

if "df_ratings" in locals():
  scheduler = GeneticScheduler(df_ratings)

  if run_trials_btn:
    st.subheader("📊 Experimental Trial Results")

    # Trial 1
    st.markdown(
        f"### Trial 1: CO_R = {co_r}, MUT_R = {mut_r} (User Configuration)"
    )
    df_1, fit_1 = scheduler.run(co_r, mut_r)
    st.success(f"Fitness Score: **{fit_1}**")
    st.dataframe(df_1, use_container_width=True)

    # Trial 2
    st.markdown("### Trial 2: CO_R = 0.90, MUT_R = 0.01 (High Exploration)")
    df_2, fit_2 = scheduler.run(0.90, 0.01)
    st.success(f"Fitness Score: **{fit_2}**")
    st.dataframe(df_2, use_container_width=True)

    # Trial 3
    st.markdown("### Trial 3: CO_R = 0.60, MUT_R = 0.05 (High Diversity)")
    df_3, fit_3 = scheduler.run(0.60, 0.05)
    st.success(f"Fitness Score: **{fit_3}**")
    st.dataframe(df_3, use_container_width=True)

    st.divider()
    st.subheader("📈 Summary Table")
    summary_df = pd.DataFrame({
        "Trial": ["Trial 1", "Trial 2", "Trial 3"],
        "CO_R": [co_r, 0.90, 0.60],
        "MUT_R": [mut_r, 0.01, 0.05],
        "Fitness Score": [fit_1, fit_2, fit_3],
    })
    st.dataframe(summary_df, use_container_width=True)
  else:
    st.info(
        "👆 Set your parameters in the sidebar and click **'Run 3 Experimental"
        " Trials'**."
    )
