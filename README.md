
# 🧠 Data-Driven, Stats-Based Football Scouting
*A Data Science Approach to Tactical Player Recommendation*

## 📄 Description

This project implements a **data-driven scouting system** for football players using advanced **statistical analysis** and **clustering techniques**. The goal is to assist scouts, analysts, and coaches in identifying optimal players for different **tactical styles** such as:

- **Tiki-Taka**
- **Counter-Attack**
- **Park the Bus**
- **Gegenpressing**
- **Total Football**

The system analyzes players' performance metrics and recommends an ideal **11-player squad** based on the selected tactic, using **K-Means clustering**, **position classification**, and **feature prioritization**.

## 🔍 Features

- 📊 Reads and processes real player statistics from the 2022–2023 season  
- 🔧 Cleans and simplifies positional data  
- 🤖 Uses **K-Means clustering** per position group (GK, DF, MF, FW)  
- 🧠 Computes **optimal number of clusters** using silhouette score  
- 🎯 Scores and ranks players based on tactical priorities  
- 📋 Selects best 11 players for a chosen tactic and formation  
- 🧬 Visualizes player clusters and team formation on a football pitch  
- 📉 Shows detailed stats of the final squad

## 🧪 Technologies Used

- Python 3  
- NumPy, Pandas  
- Scikit-learn (for clustering & scaling)  
- Matplotlib & Seaborn (for visualizations)

## 🗂️ File Structure

```
📦 Data-Driven-Stats-Based-Football-Scouting
├── final DSE project.ipynb        <- Main Jupyter notebook
├── 2022-2023 Football Stats.xlsx  <- Input player stats (external file, not uploaded)
├── README.md                      <- Project documentation
```

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+  
- Jupyter Notebook / VSCode  
- Libraries:  
  ```
  pip install numpy pandas scikit-learn matplotlib seaborn openpyxl
  ```

### ▶️ Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Data-Driven-Stats-Based-Football-Scouting.git
   cd Data-Driven-Stats-Based-Football-Scouting
   ```

2. Place your Excel file (if not included):  
   `2022-2023 Football Player Stats (1).xlsx`

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Open and run: `final DSE project.ipynb`

## 📷 Sample Output

- ✅ Visualizations of player clusters per position  
- ✅ Tactical team formation plotted on a pitch  
- ✅ Printed team list with player names, positions, and clubs  

## 📌 Use Cases

- Scouting support for football clubs and analysts  
- Data science applications in sports  
- Tactical comparison between playing styles  
- Educational resource for sports analytics

## 🙋‍♂️ Author

**Prasiddha Shukla**  
B.Tech (Electrical) | MIT-WPU | Football Enthusiast & Data Analyst
