from infrastructure.database import db
import pandas as pd
import subprocess
import sys
import threading
is_training = False

def run_retrain_async():
    thread = threading.Thread(target=merge_and_retrain)
    thread.start()

def merge_and_retrain():
    global is_training
    if is_training:
        print("⏳ Training already running...")
        return

    is_training = True
    try:
      print("🚀 Starting merge + retrain...")

      # 1️⃣ Load old data
      old_data = list(db.merged_NFR_cleaned_no_dots.find({}, {"_id": 0}))

      # 2️⃣ Load new feedback
      new_data = list(db.new_nfr_confirmed.find({}, {"_id": 0}))

      if not new_data:
        print("⚠️ No new data to merge")
        is_training = False
        return

    # 3️⃣ Convert to DataFrame
      df_old = pd.DataFrame(old_data)
      df_new = pd.DataFrame(new_data)
      df_new = df_new.rename(columns={
      "requirement": "Requirement",
      "type": "Type",
      "level": "Level"
      })
      # 4️⃣ Merge
      df_all = pd.concat([df_old, df_new], ignore_index=True)

# 🚫 remove duplicates
      df_all = df_all.drop_duplicates(
          subset=["Requirement", "Type", "Level"],
          keep="last"
      )

      # 7️⃣ Retrain model
      print("🤖 Retraining model...")

      subprocess.run([sys.executable, "-m", "ai.training.train_type_level"], check=True)

      print("✅ Retraining done!")
      print("📦 Merging data after successful training...")

      # load old + new
      old_data = list(db.merged_NFR_cleaned_no_dots.find({}, {"_id": 0}))
      new_data = list(db.new_nfr_confirmed.find({}, {"_id": 0}))

      df_old = pd.DataFrame(old_data)
      df_new = pd.DataFrame(new_data)

      df_new = df_new.rename(columns={
    "requirement": "Requirement",
    "type": "Type",
    "level": "Level"
     })

      df_all = pd.concat([df_old, df_new], ignore_index=True)

# save merged
      db.merged_NFR_cleaned_no_dots.delete_many({})
      db.merged_NFR_cleaned_no_dots.insert_many(df_all.to_dict("records"))

      print(f"✅ Merged {len(df_new)} new records")
      db.new_nfr_confirmed.delete_many({})
      print("🧹 Cleared new_nfr_confirmed")
      
    finally:
        is_training = False