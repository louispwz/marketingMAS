import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.customize_running import center_running
import pandas as pd
import numpy as np
import random 
import ydata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt


@st.cache_data(show_spinner=True)
def load_data():
    table_full = pd.read_csv("table_full.csv")
    table_vehicule = pd.read_csv("table_vehicule.csv")
    
    import json
    list_options = set()
    for value in table_vehicule['OFFER_EQUIPMENTS_MAIN_LIST'].dropna():
        try:
            for option in json.loads(value):
                list_options.add(option['label'])
        except:
            continue

    for option in list_options:
        clean_label = option.lower().replace(" ", "_").replace("/", "_").replace('é', 'e').replace('è', 'e').replace('à', 'a').replace('ù', 'u')
        flag_name = f'flag_{clean_label}'
        table_vehicule[flag_name] = table_vehicule['OFFER_EQUIPMENTS_MAIN_LIST'].apply(
            lambda x: 1 if isinstance(x, str) and option in x else 0
        )

    return table_full, table_vehicule

table_full, table_vehicule = load_data()
def bienvenue():
    st.subheader("Bienvenue sur l'application d'analyse du profil des clients Aramisauto")

    st.markdown("<br>", unsafe_allow_html=True)

    # Ajout logo
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("logo_aramisauto.jpg", width=300)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contexte et objectifs
    st.markdown(
        """
        <div style="font-size: 20px; line-height:1.4;">

        **Contexte :**  

        Aramisauto souhaite mieux comprendre le profil de ses clients pour optimiser sa stratégie marketing.  
        Nous analysons des données internes, enrichies avec des sources externes, pour obtenir une vue complète du comportement de nos clients.



        **Objectifs du projet :**

        - **Analyser les données internes** pour identifier les comportements et les préférences des clients.
        - **Enrichir les données** avec des informations externes afin d'avoir une compréhension globale.
        - **Développer une appli interactive** pour visualiser et interpréter ces données.
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Instructions de navigation
    st.markdown("""
        <div style="font-size: 20px; line-height:1.4;">

        **Navigation :**  
        attendre que ça soit fini mais en gros on parle des hypothèses
        </div>
        """,
    unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Information complémentaire
    st.info("Cette application a été développée dans le cadre du projet Datamining pour l'année 2024/2025.")
    
    
    # st.subheader("Statistiques descriptives")
    # st.write(table_full.describe(include='all'))
    
    
    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3, animation_length=600)

bienvenue.__name__ = "Accueil" # change le nom dans le sidebar


    
def ensemble():
    st.title("Statistiques descriptives")

    #metrics clients
    st.markdown("<h1 style='text-decoration: underline;'>Statistiques clients</h1>", unsafe_allow_html=True)
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    
    col1_1.metric("Âge moyen des clients", f"{table_full['age_client'].mean():.1f} ans")
    col1_2.metric("Ancienneté moyenne", f"{table_full['anciennete'].mean():.1f} ans")
    col1_3.metric("Nombre de femmes", table_full['GENDER'].value_counts().get('F', 0))
    col1_4.metric("Nombre d'hommes", table_full['GENDER'].value_counts().get('H', 0))

    # Graphiques clients
    col_client_1, col_client_2 = st.columns(2)

    with col_client_1.container():
        st.subheader("Répartition par tranches d'âge")
        fig1, ax1 = plt.subplots()
        ax1.hist(table_full['age_client'].dropna(), bins=10, color=color1, edgecolor='black')
        ax1.set_xlabel("Âge")
        ax1.set_ylabel("Nombre de clients")
        ax1.set_title("Distribution de l'âge des clients")
        st.pyplot(fig1)

    with col_client_2.container():
        st.subheader("Répartition par genre")
        fig2, ax2 = plt.subplots()
        gender_counts = table_full['GENDER'].value_counts()
        ax2.pie(gender_counts, labels=["Hommes", "Femmes"], autopct='%1.1f%%', startangle=90, 
            colors=[color1, color2], wedgeprops={'edgecolor': 'black'})
        ax2.axis('equal')
        st.pyplot(fig2)

    
    #metrics voitures
    st.markdown("<h1 style='text-decoration: underline;'>Statistiques voitures</h1>", unsafe_allow_html=True)
    
    col2_1, col2_2, col2_3 = st.columns(3)

    col2_1.metric("Nombre de voitures disponibles :", table_vehicule.shape[0])
    col2_2.metric("Nombre de modèles de voitures :", table_vehicule['VEHICULE_MODELE'].nunique())
    col2_3.metric("Nombre de marques différentes disponibles :", table_vehicule["VEHICULE_MARQUE"].nunique())
    
    col3_1, col3_2, col3_3 = st.columns(3)
    
    col3_1.metric("Taux de reprise", f"{(table_full['FLAG_REPRISE'].sum() / table_full['FLAG_COMMANDE'].sum() * 100):.1f} %")
    col3_2.metric("Kilométrage moyen", f"{table_vehicule['VEHICULE_KM'].mean():,.0f} km")    
    col3_3.metric("Prix moyen TTC", f"{table_full['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")

    
    col_voitures_1, col_voitures_2 = st.columns(2)
    #repartition par marque
    with col_voitures_1.container():
        st.subheader("Nombre de voitures par marque")
        qtte_marque = st.number_input("Nombre de marques à représenter", min_value=1, max_value=table_vehicule['VEHICULE_MARQUE'].nunique(), value=10)
        fig5, ax5 = plt.subplots()
        top_marques = table_vehicule['VEHICULE_MARQUE'].value_counts().head(qtte_marque)
        ax5.barh(top_marques.index[::-1], top_marques.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax5.set_xlabel("Nombre de voitures")
        ax5.set_ylabel("Marque")
        st.pyplot(fig5)
        
    #repartion par modele
    with col_voitures_2.container():
        st.subheader("Nombre de voitures par modèle")
        qtte_modele = st.number_input("Nombre de modèles à représenter", min_value=1, max_value=table_vehicule['VEHICULE_MODELE'].nunique(), value=10)
        fig7, ax7 = plt.subplots()
        top_modeles = table_vehicule['VEHICULE_MODELE'].value_counts().head(qtte_modele)
        ax7.barh(top_modeles.index[::-1], top_modeles.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax7.set_xlabel("Nombre de voitures")
        ax7.set_ylabel("Modèle")
        st.pyplot(fig7)
    

    col_voitures_2_1, col_voitures_2_2 = st.columns(2)
    # VO vs VN
    with col_voitures_2_1.container():
        st.subheader("Répartition des voitures neuves / d'occasions")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        table_vehicule['VEHICULE_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax6, colors=[color1, color2],wedgeprops={'edgecolor': 'black'})
        ax6.set_ylabel('')
        st.pyplot(fig6)

    with col_voitures_2_2.container():
        st.subheader("Prix de vente des véhicules")
        nb_bins = st.slider("Nombre de bins", 0, 30, 20)
        fig4, ax4 = plt.subplots()
        prix_vente = table_full['PRIX_VENTE_TTC_COMMANDE'].dropna()
        ax4.hist(prix_vente, bins=nb_bins, color=color1,edgecolor='black')
        ax4.set_xlabel("Prix de vente TTC")
        ax4.set_ylabel("Quantité")
        st.pyplot(fig4)
        
    col_voitures_3_1, col_voitures_3_2 = st.columns(2)
    #repartition par carburant
    with col_voitures_3_1.container():
        energy_colors = {'Essence': "#2ecc71",'Diesel': "#FDDB15",'Hybride': "#9900D0",'Electrique': "#2980b9",'Autre': "#A4A4A4"}
        st.subheader("Répartition des carburants")
        fig3, ax3 = plt.subplots()
        top_energies = table_vehicule['VEHICULE_ENERGIE'].value_counts()
        colors = [energy_colors.get(energy, "#bdc3c7") for energy in top_energies.index]
        ax3.barh(top_energies.index[::-1], top_energies.values[::-1], color=plt.cm.Purples(np.linspace(0.2, 0.9, 10)))
        ax3.set_xlabel("Nombre de voitures")
        ax3.set_ylabel("Carburant")
        st.pyplot(fig3)

    
    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)
        
ensemble.__name__ = "Statistiques descriptives"       
        
def page_1():
    st.title("Hypothèse Prix X Options")
    st.write("Dans cette hypothèse, nous cherchons à déterminer si le nombre d’options ou d’équipements disponibles sur un modèle de véhicule a un impact significatif sur son prix de vente.")
    st.write("L'objectif est d'étudier s'il existe une corrélation entre le niveau d’équipement d'un véhicule et son positionnement tarifaire. En d’autres termes, plus un véhicule possède d’options (comme la climatisation, le GPS, le radar de recul, etc.), plus son prix final est susceptible d'être élevé. Cette analyse permettrait de mieux comprendre le rôle des équipements dans la valorisation commerciale des véhicules et d’identifier dans quelle mesure ils influencent la stratégie de tarification.")
    #tout pour pouvoir travailler comme il faut
    EQUIPEMENTS = ['flag_jantes_alliage','flag_toit_ouvrant_panoramique','flag_climatisation','flag_regulateur_de_vitesse','flag_radar_de_recul','flag_gps','flag_camera_de_recul','flag_interieur_cuir','flag_bluetooth','flag_apple_car_play','flag_android_auto']
    table_vehicule['NB_EQUIPEMENTS'] = table_vehicule[EQUIPEMENTS].sum(axis=1)
    
    table_merged = table_full.merge(table_vehicule,how="left",left_on="VEHICULE_ID_COMMANDE",right_on="VEHICULE_ID")    
    
    st.subheader("Prix selon le niveau d’équipement")
    col_1, col_2, col_3 = st.columns(3)
    nb_equip = col_3.slider("nombre d'équipement", 0, table_merged['NB_EQUIPEMENTS'].max(), 5)
    
    #metric
    col_1,col_2, col_3 = st.columns(3)
    with col_1.container():
        st.metric("Prix moyen (0 équipement)",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == 0]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
    with col_2.container():
        st.metric("Prix moyen (max équipements)",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == table_merged['NB_EQUIPEMENTS'].max()]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
    with col_3.container():
        st.metric(f"Prix moyen ({nb_equip} équipement)",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == nb_equip]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
        
    
    col_1, col_2 = st.columns(2)
    #prix en fonction du nombre d'equip / scatterplot
    with col_1.container():
        fig8, ax8 = plt.subplots()
        ax8.scatter(table_merged["NB_EQUIPEMENTS"],table_merged["PRIX_VENTE_TTC_COMMANDE"],alpha=0.5, color=color1, edgecolors='black')
        ax8.set_xlabel("Nombre d'équipements")
        ax8.set_ylabel("Prix TTC en €")
        ax8.set_title("Corrélation entre prix et nombre d'équipements")
        st.pyplot(fig8)
    
    #prix moyen en fonction du nombre d'equip
    with col_2.container():
        mean_prix = table_merged.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()
        fig9, ax9 = plt.subplots()
        ax9.plot(mean_prix["NB_EQUIPEMENTS"],mean_prix["PRIX_VENTE_TTC_COMMANDE"],marker='o',color=color1)
        ax9.set_xlabel("Nombre d'équipements")
        ax9.set_ylabel("Prix moyen TTC en €")
        ax9.set_title("Prix moyen par nombre d'équipements")
        st.pyplot(fig9)
        
    with st.container():
        col_1, col_2 = st.columns(2)    
        
        col_1.write('Ce graphique montre la relation entre le nombre d’équipements présents dans un véhicule et son prix TTC. On observe une tendance générale où le prix augmente avec le nombre d’équipements. Les points sont dispersés, mais on remarque clairement des "paliers" pour chaque nombre d’équipements, indiquant que les voitures mieux équipées tendent à être plus chères')
        col_2.write("Ce graphique présente l'évolution du prix moyen TTC en fonction du nombre d’équipements. La courbe montre une progression régulière, avec une accélération notable à partir de 8 équipements. Cela confirme que plus un véhicule dispose d’équipements, plus son prix moyen est élevé, avec un effet particulièrement marqué pour les véhicules très bien équipés.")
    col_1, col_2 = st.columns(2)
    
    
    #prix par marque en fct du nmobre d'équipement
    with col_1.container():
        st.subheader("Prix moyen selon nombre d’équipements, par marque")
        #choix marque 
        marques = table_merged['VEHICULE_MARQUE_x'].dropna().unique()
        selected_marque = st.selectbox("Marque : ", sorted(marques))

        table_merged_marque = table_merged[table_merged['VEHICULE_MARQUE_x'] == selected_marque]
        prix_equip = table_merged_marque.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()

        fig10, ax10 = plt.subplots()
        ax10.plot(prix_equip["NB_EQUIPEMENTS"],prix_equip["PRIX_VENTE_TTC_COMMANDE"],marker='o',linestyle='-',color=color1)
        ax10.set_xlabel("Nombre d'équipements")
        ax10.set_ylabel("Prix moyen TTC en €")
        ax10.set_title(f"Prix moyen par nombre d’équipements de la marque {selected_marque}")
        st.pyplot(fig10)
        
        
        
    #prix par modele en fct du nmobre d'équipement
    with col_2.container():
        st.subheader("Prix moyen selon nombre d’équipements, par modele")
        #choix modele
        col1, col2 = st.columns(2)
        marques = table_merged['VEHICULE_MARQUE_x'].dropna().unique()
        selected_marque = col1.selectbox("Marque :", sorted(marques))
        modeles = table_merged[table_merged['VEHICULE_MARQUE_x'] == selected_marque]['VEHICULE_MODELE_x'].dropna().unique()
        selected_modele = col2.selectbox("Modèle :", sorted(modeles))

        #filtre df
        table_merged_modele = table_merged[(table_merged['VEHICULE_MARQUE_x'] == selected_marque) &(table_merged['VEHICULE_MODELE_x'] == selected_modele)]
        prix_equip = table_merged_modele.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()


        fig11, ax11 = plt.subplots()
        ax11.plot(prix_equip["NB_EQUIPEMENTS"],prix_equip["PRIX_VENTE_TTC_COMMANDE"],marker='o',linestyle='-',color=color1)
        ax11.set_xlabel("Nombre d'équipements")
        ax11.set_ylabel("Prix moyen TTC en €")
        ax11.set_title(f"Prix moyen par nombre d’équipements du modele {selected_modele}")
        st.pyplot(fig11)

        
    st.subheader("Analyse en Composantes Principales des Équipements")
    
    flag_cols = [col for col in table_merged.columns if col.startswith("flag_") and col.endswith("_y")]
    X = table_merged[flag_cols].copy()
    X.columns = [col.replace("_y", "") for col in X.columns]
    df = table_merged.dropna(subset=["PRIX_VENTE_TTC_COMMANDE"])
    X = X.loc[df.index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["PRIX"] = df["PRIX_VENTE_TTC_COMMANDE"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["PRIX"], cmap="viridis", alpha=0.6)
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Prix TTC")
    st.pyplot(fig)
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)
    top_features_pc1 = loadings.abs().sort_values("PC1", ascending=False)
    st.subheader("Equipement les plus contributifs au prix")
    st.dataframe(top_features_pc1)
    correlation = np.corrcoef(df_pca["PC1"], df_pca["PRIX"])[0, 1]
    st.markdown(f"**Corrélation entre PC1 et le prix :** `{correlation:.2f}`")
    st.write("La corrélation de 0.47 entre la première composante de l’ACP et le prix montre qu’il existe un lien positif entre les équipements et le prix des voitures. Cela signifie que, globalement, plus une voiture est équipée, plus elle coûte cher. Ce n’est pas une relation parfaite, mais elle est assez claire pour dire que les équipements ont un vrai impact sur la valeur des véhicules.")

    st.subheader("Conclusion de l'hypothèse")
    st.write("On observe que plus une voiture est équipée, plus son prix augmente. Toutes les options n’ont pas le même impact : certaines, comme le bluetooth ou la climatisation, sont aujourd’hui très répandues et n’influencent plus autant le prix qu’auparavant. En revanche, Android auto, apple carplay ou le radar de recul ont un effet plus fort sur la prise en valeur du véhicule. Cela dit, le nombre total d’équipements reste un bon indicateur de l'évolution du prix d’une voiture. Même si certaines options sont devenues la norme, leur accumulation justifie souvent un prix plus élevé.")

    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)
    
        
page_1.__name__ = "Hypothèse prix X options"
    
    
def page_2():
    st.title("Nom Hypothèse")
    st.write("description de l'hypothèse vite fait")
    st.write("graphiques/tableaux + interprétations")
    st.write("conclu de l'hypothèse")


page_2.__name__ = "Hypothèse 2"
            
def page_3():
    st.title("Nom Hypothèse")
    st.write("description de l'hypothèse vite fait")
    st.write("graphiques/tableaux + interprétations")
    st.write("conclu de l'hypothèse")

page_3.__name__ = "Hypothèse 3"

def page_4():
    st.title("Nom Hypothèse")
    st.write("description de l'hypothèse vite fait")
    st.write("graphiques/tableaux + interprétations")
    st.write("conclu de l'hypothèse")

page_4.__name__ = "Hypothèse 4"


st.set_page_config(
    page_title="Analyse du profil des clients Aramisauto",
    page_icon="🚗",
    layout="wide")


pg = st.navigation({
    "Analyse du profil des clients Aramisauto" : [bienvenue, ensemble, page_1, page_2, page_3, page_4]
})

switch_voiture = st.sidebar.toggle("Activer le mode voiture")


color1 = st.sidebar.color_picker("Couleur 1", "#89CFF0")
color2 = st.sidebar.color_picker("Couleur 2", "#B19CD9") 


st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #999; padding-top: 1rem;">
       Développé par BRAULT Juliette, CAUSEUR Léna et PRUSIEWICZ Louis.
    </div>
    """,
    unsafe_allow_html=True
)

pg.run()


