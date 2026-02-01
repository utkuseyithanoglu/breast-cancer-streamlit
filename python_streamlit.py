import streamlit as st
import pandas as pd
import os
import pickle
st.set_page_config(page_title='Breast Cancer Demo', layout='wide')
st.title('Breast Cancer Predict Demo')
patient_CSV = "patients.csv"
results_CSV = "results.csv"
if not os.path.exists(patient_CSV):
    pd.DataFrame(columns=['name', 'last4']).to_csv(patient_CSV, index=False)
if not os.path.exists(results_CSV):
    pd.DataFrame(columns=['name','last4','mean_area','worst_radius','worst_perimeter',
                          'worst_concave_points','mean_concave_points ']).to_csv(results_CSV, index=False)
if "show_doctor" not in st.session_state:
    st.session_state.show_doctor = False
if "show_register" not in st.session_state:
    st.session_state.show_register = False
if "users" not in st.session_state:
    st.session_state.users = {}
if "patients" not in st.session_state:
    st.session_state.patients = {}
if st.sidebar.button('sign up'):
    st.session_state.show_register = True
    st.session_state.show_doctor = False
if st.session_state.show_register:
    st.subheader("Patient Register")
    with st.form("register_form"):
        user_name = st.text_input("Enter your username")
        last4 = st.text_input("Enter your last 4 digits on phone number", max_chars=4)
        submit = st.form_submit_button("register patient")
    if submit:
        if user_name and last4.isnumeric() and len(last4) == 4:
            key = f"{user_name}-{last4}"
            st.session_state.users[key] = {"name": user_name, "last4": last4}
            patient_df = pd.read_csv(patient_CSV)
            patient_df = pd.concat([patient_df, pd.DataFrame([{
                    "name": user_name.strip(),
                    "last4": last4.strip() }])],ignore_index=True)
            patient_df.to_csv(patient_CSV, index=False)
            st.session_state.show_register = False
            st.success("Patient Registered")
        else:
            st.error("Invalid Username or Last4")
if st.sidebar.button('doctor login'):
    st.session_state.show_doctor = True
    st.session_state.show_register = False

if st.session_state.show_doctor:
    st.subheader("Doctor Login")

    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False
    with st.form("login_form"):
        st.info('this is a demo version please write utku')
        doctor_id = st.text_input("Doctor ID")
        login_form = st.form_submit_button("login")

    if login_form and doctor_id == "utku":
        st.session_state.doctor_logged_in = True
    elif login_form and doctor_id != "utku":
        st.error("Invalid Doctor ID")

    if st.session_state.doctor_logged_in:

        patient_df = pd.read_csv(patient_CSV)
        patient_df["display"] = patient_df["name"] + " - " + patient_df["last4"].astype(str)

        selected = st.selectbox("Select patient", patient_df["display"], key="selected_patient")
        sel_row = patient_df.loc[patient_df["display"] == selected].iloc[0]

        selected_name = sel_row["name"]
        selected_last4 = str(sel_row["last4"])

        st.subheader("Provide information")

        with st.form("feature_form"):
            worst_perimeter = st.number_input("worst perimeter", key="worst_perimeter")
            st.image("assets/worst_perimeter1.png", width=400)

            worst_concave_points = st.number_input("worst concave points", key="worst_concave_points")
            st.image("assets/worst_concave1.png", width=400)

            mean_concave_points = st.number_input("mean concave points", key="mean_concave_points")
            st.image("assets/mean_concave_points1.png", width=400)

            worst_radius = st.number_input("worst radius", key="worst_radius")
            st.image("assets/worst_radius1.png", width=400)

            mean_area = st.number_input("mean area", key="mean_area")
            st.image("assets/mean_area1.png", width=400)

            feature_submit = st.form_submit_button("save features")

        if feature_submit:
            @st.cache_resource
            def load_model():
                with open("cancer_rf_best.pkl", "rb") as m:
                    return pickle.load(m)

            model = load_model()

            X = [[mean_area, worst_radius, worst_perimeter, worst_concave_points, mean_concave_points]]
            pred = int(model.predict(X)[0])
            results_df = pd.read_csv(results_CSV)

            new_row = {
                "name": selected_name,
                "last4": selected_last4,
                "mean_area": mean_area,
                "worst_radius": worst_radius,
                "worst_perimeter": worst_perimeter,
                "worst_concave_points": worst_concave_points,
                "mean_concave_points": mean_concave_points,
                "prediction": pred
            }

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(results_CSV, index=False)

            st.success("Saved to results.csv ✅")

            st.success("Prediction calculated ")
            st.write("Prediction:", "Healthy  " if pred == 0 else "Unhealthy ")

            st.write("Raw pred:", pred)

if "show_patient_login" not in st.session_state:
    st.session_state.show_patient_login = False

if st.sidebar.button("login"):
    st.session_state.show_doctor = False
    st.session_state.show_register = False
    st.session_state.show_patient_login = True
    st.info('you can use(name :tryout last4:1234)')

if st.session_state.show_patient_login:

    with st.form("patient_form"):
        fname = st.text_input("Patient Name")
        flast4 = st.text_input("Patient Last 4", max_chars=4)
        login_form = st.form_submit_button("login")

    if login_form:
        results_df = pd.read_csv(results_CSV)

        results_df["name"]  = results_df["name"].astype(str).str.strip()
        results_df["last4"] = results_df["last4"].astype(str).str.strip()

        fname  = str(fname).strip()
        flast4 = str(flast4).strip()

        row = results_df[(results_df["name"] == fname) & (results_df["last4"] == flast4)]

        if row.empty:
            st.error("no user  / name + last4 isint cacth ")
        else:
            st.success("login is successful ")
            st.dataframe(row)
            st.write('average mean concave points : 0.04891914586994728')
            st.write('worst concave points : 0.11460622319859401')
            st.write('data mean(mean area) : 654.8891036906855')
            st.write('data mean worst radius : 16.269189806678387')
            st.write('data worst perimeter: 107.26121265377857')
            pred = row['prediction'].iloc[0]

            if pred == 0:
                st.success("Result: HEALTHY")
            else:
                st.error("Result: UNHEALTHY")























