import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration des graphiques
try:
    plt.style.use('seaborn')
except:
    pass  # Utiliser le style par défaut si seaborn nn 'est pas disponible
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def load_data(file_path):
    """Charger les données depuis le CSV"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Données chargées avec succès. {df.shape[0]} lignes et {df.shape[1]} colonnes.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def create_age_groups(df):
    """Créer des tranches d'âge pour l'analyse"""
    # Copier le dataframe pour éviter les warnings
    df_copy = df.copy()
    
    # Vérifier si nous avons déjà une colonne âge, sinon la calculer
    if 'age_client' in df.columns:
        # Créer des tranches d'âge
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
        df_copy['tranche_age'] = pd.cut(df_copy['age_client'], bins=bins, labels=labels, right=False)
    else:
        print("La colonne 'age_client' n'est pas disponible. Impossible de créer des tranches d'âge.")
    
    return df_copy

def standardize_colors(df):
    """Standardiser les couleurs en groupes plus génériques en utilisant regex"""
    df = df.copy()
    
    # Vérifier que la colonne existe
    if 'VEHICULE_COULEUR' not in df.columns:
        print("Colonne VEHICULE_COULEUR manquante.")
        return df
    
    # Créer un dictionnaire de mapping avec des expressions régulières
    color_mapping = {
        'NOIR': r'noir|black|anthracite|carbone|graphite|Saphirschwarz|schwarz|nero|negro|onyx|jet|obsidian|carbon|ebony|raven|charcoal|nera|carbonscwarz|night|midnightblack|blackmetallic|umbra|mokka|Brun Adventure',
        
        'BLANC': r'blanc|white|nacré|perle|ivory|Bianco|Alpinweiss|Ivoire|weiss|weiß|bianche|blanche|polar|crystal|snowflake|glacier|diamantweiß|perlweiß|calcite|pure|mineralweiß|alaska|perlmutt|cristal\s*pearl|crystalline|ice|frost|snow|diamond|platinweiß|alpine|oyster|shell|parchment|cream|eggshell|magnolia',
        
        'GRIS': r'gris|grey|gray|metal|argent|silver|Acier Moderne|Grau|mondsteingrau|Grigio|silber|plata|platine|steel|ash|palladium|selenite|quartz|rhodium|aluminium|titanium|eissilber|tundragrau|glaciersilver|granit|florett|iridium|selenit|manhattan|indium|tenorit|platin|stonegrey|artense\s*grijs| Artense Grijs|Grafietgrijs|Berggrijs',
        
        'BLEU': r'bleu|blue|azur|marine|turquoise|cyan|Blu|Mediterranblau|azzurre|ozean|atlantis|pacific|adriatic|fjord|windsor|moonlight|portimao|tanzanite|reef|topaz|sky|cobalt|navarra|estoril|mystic|utopia|petrol|sapphire|amalfi|midnight|aqua|capri|imperialblau|dive\s*in\s*jeju|phytonicblau|phytonic| Imperialblau Brillanteffekt| Dive in Jeju|Phytonicblau|Blau|Nautile|AZUL',
        
        'ROUGE': r'rouge|red|bordeaux|carmin|rubis|burgundy|rosso|rot|rojo|cardinal|magma|crimson|tango|lava|vulcano|tornado|cranberry|corsa|carmijn|misano|imola|melbourne|indianapolis|monza|sakhir|firenze|scala|passion|fiammante',
        
        'VERT': r'vert|green|emeraude|olive|kaki|grün|verde|oliva|british|racing|forest|jade|sage|mint|aventurine|goodwood|malachite|hunter|alpine|rallye|mint|serpentino|pino|wilderness|jungle',
        
        'JAUNE': r'jaune|yellow|or|gold|citron|Sable \(M0EU\)|Sable|gelb|amarillo|giallo|soleil|vegas|aspen|bahama|sunburst|arizona|dakar|austin|phoenix|speed|atacama|modena|sunrise|sandstone',
        
        'MARRON': r'marron|brown|chocolat|beige|camel|taupe|bronze|Brun Terracotta \(CNZ\)|Brun Terracotta|braun|marrón|marrone|mocha|tobacco|cognac|malt|havana|teaktree|mocca|mahogany|walnut|saddle|terra|kastanien|havanna|sepang|cappuccino|sahara|atacama|vison|brun\s*vison|oak|teak|suede|espresso|sienna|hickory|cashew|cinnamon|pecan|buff|tan|sepia|umber|brun\s*moka|moka|Brun Moka',
        
        'ORANGE': r'orange|amber|cuivre|copper|arancione|naranja|valencia|coral|mandarin|phoenix|sunrise|sunset|atomic|burnt|fusion|glowing|volcanic|flame|fiery|tiger|bronze|canyon',
        
        'VIOLET': r'violet|purple|mauve|lilas|lavande|violett|púrpura|viola|ultraviolet|amethyst|plum|aubergine|twilight|velvet|mystic|passion|eggplant|cassis|améthyste|amethyste|\(9AH\)|mulberry|heather|orchid|grape|byzantium|thistle',
        
        'ROSE': r'rose|pink|fuchsia|magenta|rosa|shocking|flamingo|bubblegum|raspberry|cerise|candy|orchid'
    }
    
    # Appliquer le mapping aux couleurs existantes
    df['COULEUR_STANDARD'] = df['VEHICULE_COULEUR'].str.lower()
    
    for standard_color, regex_pattern in color_mapping.items():
        mask = df['COULEUR_STANDARD'].str.contains(regex_pattern, case=False, regex=True, na=False)
        df.loc[mask, 'COULEUR_STANDARD'] = standard_color
    
    # Pour les couleurs qui n'ont pas été mappées, utiliser 'AUTRE'
    mapped_colors = list(color_mapping.keys())
    df.loc[~df['COULEUR_STANDARD'].isin(mapped_colors), 'COULEUR_STANDARD'] = 'AUTRE'
    
    # Afficher un résumé du mapping
    print("Résumé du mapping des couleurs:")
    for std_color in sorted(df['COULEUR_STANDARD'].unique()):
        original_colors = df[df['COULEUR_STANDARD'] == std_color]['VEHICULE_COULEUR'].unique()
        if len(original_colors) > 5:
            print(f"{std_color}: {', '.join(original_colors[:5])}... ({len(original_colors)} couleurs)")
        else:
            print(f"{std_color}: {', '.join(original_colors)}")
    
    return df

import plotly.express as px
import plotly.graph_objects as go

def analyze_colors_by_age(df):
    """Analyser les couleurs préférées par tranche d'âge avec des graphiques interactifs"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_commandes = df[df['FLAG_COMMANDE'] == 1].copy()

    # Vérifier que les colonnes nécessaires existent
    if 'COULEUR_STANDARD' not in df_commandes.columns or 'tranche_age' not in df_commandes.columns:
        print("Colonnes requises manquantes.")
        return None, None, None

    # 1. Tableau croisé des couleurs par tranche d'âge
    crosstab = pd.crosstab(df_commandes['tranche_age'], df_commandes['COULEUR_STANDARD'])

    # Normaliser par ligne pour avoir des pourcentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    # 2. Heatmap interactive avec Plotly
    fig_heatmap = px.imshow(
        crosstab_pct,
        labels=dict(x="Couleur du véhicule", y="Tranche d'âge", color="Pourcentage"),
        x=crosstab_pct.columns,
        y=crosstab_pct.index,
        text_auto=".1f"  # Afficher les pourcentages avec une décimale
    )
    fig_heatmap.update_layout(
        title="Pourcentage de chaque couleur de véhicule par tranche d'âge",
        xaxis_title="Couleur du véhicule",
        yaxis_title="Tranche d'âge",
        coloraxis_colorbar=dict(title="Pourcentage (%)"),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig_heatmap.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })

    fig_heatmap.update_coloraxes(showscale=False)

    """Analyser les couleurs préférées par tranche d'âge avec des graphiques interactifs dans une seule fenêtre"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_commandes = df[df['FLAG_COMMANDE'] == 1].copy()

    # Vérifier que les colonnes nécessaires existent
    if 'COULEUR_STANDARD' not in df_commandes.columns or 'tranche_age' not in df_commandes.columns:
        print("Colonnes requises manquantes.")
        return None

    # Obtenir les tranches d'âge uniques
    age_groups = sorted(df_commandes['tranche_age'].unique())

    # Définir un dictionnaire de correspondance entre les noms de couleurs et les codes hexadécimaux
    color_mapping = {
        'NOIR': '#000000',     # Noir
        'BLANC': '#FFFFFF',    # Blanc
        'GRIS': '#A9A9A9',     # Gris
        'BLEU': '#0000FF',     # Bleu
        'ROUGE': '#FF0000',    # Rouge
        'VERT': '#008000',     # Vert
        'JAUNE': '#FFD700',    # Jaune doré
        'MARRON': '#8B4513',   # Marron
        'ORANGE': '#FFA500',   # Orange
        'VIOLET': '#800080',   # Violet
        'ROSE': '#FFC0CB',     # Rose
        'AUTRE': '#808080'     # Gris pour "Autre"
    }

    # Créer une figure avec des subplots
    fig = make_subplots(
        rows=2, cols=3,  # 2 lignes, 3 colonnes pour 6 tranches d'âge
        subplot_titles=[f"Tranche d'âge {age}" for age in age_groups]
    )

    # Ajouter un graphique pour chaque tranche d'âge
    for i, age_group in enumerate(age_groups):
        # Obtenir les données pour la tranche d'âge
        age_data = df_commandes[df_commandes['tranche_age'] == age_group]
        color_counts = age_data['COULEUR_STANDARD'].value_counts()

        # Appliquer les couleurs aux barres
        bar_colors = [color_mapping.get(color, '#808080') for color in color_counts.index]

        # Ajouter un barplot pour cette tranche d'âge
        fig.add_trace(
            go.Bar(
                x=color_counts.index,
                y=color_counts.values,
                name=f"Tranche d'âge {age_group}",
                marker=dict(
                    color=bar_colors,  
                    line=dict(color='black', width=1.5)  #Contour noir pour pouvoir voir les barres blanches
            )
            ),
            row=(i // 3) + 1,  # Ligne du subplot
            col=(i % 3) + 1    # Colonne du subplot
        )

    # Mettre à jour la mise en page
    fig.update_layout(
        title="Répartition des couleurs par tranche d'âge",
        height=800,  # Hauteur totale de la figure
        showlegend=False,  # Désactiver la légende globale
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })

    # Ajouter des titres aux axes
    fig.update_xaxes(title_text="Couleur", row=2, col=1)
    fig.update_yaxes(title_text="Nombre de véhicules", row=1, col=1)


    return fig_heatmap, fig, crosstab_pct

def perform_regression_analysis(df):
    """Réaliser une analyse de régression logistique pour prédire les préférences de couleur selon l'âge"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_model = df[df['FLAG_COMMANDE'] == 1].copy()
    
    # Enlever les valeurs manquantes
    df_model = df_model.dropna(subset=['COULEUR_STANDARD', 'age_client'])
    
    # Utiliser toutes les couleurs standardisées
    colors_to_analyze = df_model['COULEUR_STANDARD'].value_counts().index.tolist()
    
    # Encoder les couleurs
    le = LabelEncoder()
    df_model['color_encoded'] = le.fit_transform(df_model['COULEUR_STANDARD'])
    color_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    print("Encodage des couleurs standardisées:")
    for color, code in color_mapping.items():
        print(f"{color}: {code}")
    
    # Préparer les données pour la régression multinomiale
    X = df_model[['age_client']]
    y = df_model['color_encoded']
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Créer des variables dummies pour chaque couleur (one vs rest)
    results = {}
    
    for color, code in color_mapping.items():
        print(f"\nAnalyse de régression pour la couleur: {color}")
        
        # Créer une variable binaire pour cette couleur
        df_model[f'is_{color}'] = (df_model['COULEUR_STANDARD'] == color).astype(int)
        
        # Effectuer la régression logistique avec statsmodels pour des statistiques détaillées
        model_formula = f'is_{color} ~ age_client'
        model = smf.logit(formula=model_formula, data=df_model).fit(disp=0)
        
        # Afficher les résultats
        print(model.summary().tables[1])
        
        # Stocker les résultats
        results[color] = {
            'coefficient': model.params['age_client'],
            'p_value': model.pvalues['age_client'],
            'odds_ratio': np.exp(model.params['age_client']),
            'significant': model.pvalues['age_client'] < 0.05
        }
    
    # Créer un DataFrame des résultats pour une visualisation facile
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('coefficient', ascending=False)
    
    # Graphique interactif des coefficients de régression
    fig_coefficients = go.Figure()
    fig_coefficients.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['coefficient'],
            marker_color=['darkblue' if sig else 'lightblue' for sig in results_df['significant']],
            text=[f"p-value: {p:.3e}" for p in results_df['p_value']],
            textposition='outside'
        )
    )
    fig_coefficients.update_layout(
        title="Coefficient de l'âge sur la probabilité de choisir chaque couleur",
        xaxis_title="Couleur",
        yaxis_title="Coefficient (log-odds)",
        template="plotly_white",
        height=500
    )

    fig_coefficients.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })
    
    # Graphique interactif des odds ratios
    fig_odds_ratios = go.Figure()
    fig_odds_ratios.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['odds_ratio'],
            marker_color=['darkblue' if sig else 'lightblue' for sig in results_df['significant']],
            text=[f"p-value: {p:.3e}" for p in results_df['p_value']],
            textposition='outside'
        )
    )
    fig_odds_ratios.update_layout(
        title="Odds Ratio de l'âge sur la probabilité de choisir chaque couleur",
        xaxis_title="Couleur",
        yaxis_title="Odds Ratio",
        template="plotly_white",
        height=500
    )

    fig_odds_ratios.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })
    
    return results_df, fig_coefficients, fig_odds_ratios



def plot_color_preferences_by_age_continuous(df):
    """Créer un graphique interactif montrant l'évolution des préférences de couleurs avec l'âge (continu)"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_commandes = df[df['FLAG_COMMANDE'] == 1].copy()

    # Enlever les valeurs manquantes
    df_commandes = df_commandes.dropna(subset=['COULEUR_STANDARD', 'age_client'])

    # Définir un dictionnaire de correspondance entre les noms de couleurs et les codes couleurs
    color_mapping = {
        'NOIR': '#000000',     # Noir
        'BLANC': '#E0E0E0',    # Blanc
        'GRIS': '#A9A9A9',     # Gris
        'BLEU': '#0000FF',     # Bleu
        'ROUGE': '#FF0000',    # Rouge
        'VERT': '#008000',     # Vert
        'JAUNE': '#FFD700',    # Jaune doré
        'MARRON': '#8B4513',   # Marron
        'ORANGE': '#FFA500',   # Orange
        'VIOLET': '#800080',   # Violet
        'ROSE': '#FFC0CB',     # Rose
        'AUTRE': '#808080'     # Gris pour "Autre"
    }

    # Créer des bins d'âge plus fins pour une visualisation continue
    age_bins = np.arange(20, 80, 5)  # tranches d'âge de 5 ans
    df_commandes['age_bin'] = pd.cut(df_commandes['age_client'], bins=age_bins)

    # Calculer la distribution des couleurs pour chaque bin d'âge
    color_distribution = pd.crosstab(df_commandes['age_bin'], df_commandes['COULEUR_STANDARD'], normalize='index') * 100

    # Convertir les bins en valeurs numériques pour le graphique (point médian de chaque bin)
    age_mid_points = [(b.left + b.right) / 2 for b in color_distribution.index]

    # Créer le graphique interactif avec Plotly
    fig = go.Figure()

    # Ajouter une ligne pour chaque couleur standardisée
    for color in color_distribution.columns:
        fig.add_trace(
            go.Scatter(
                x=age_mid_points,
                y=color_distribution[color],
                mode='lines+markers',
                name=color,
                line=dict(color=color_mapping.get(color, '#808080'), width=2,dash='dash' if color == 'BLANC' else None),  # Ligne en pointillés pour le blanc
                marker=dict(size=8, symbol='circle', line=dict(color='black', width=1))  # Contour noir pour les marqueurs
            )
        )

    # Mettre à jour la mise en page
    fig.update_layout(
        title="Évolution des préférences de couleur selon l'âge",
        xaxis_title="Âge",
        yaxis_title="Pourcentage (%)",
        legend_title="Couleur",
        template="plotly_white",
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })

    return fig
def main():
    # Charger les données
    file_path = 'table_full.csv'
    df = load_data(file_path)
    
    if df is not None:
        # Standardiser les couleurs
        df = standardize_colors(df)
        
        # Créer les tranches d'âge
        df = create_age_groups(df)
        
        # Analyser les couleurs par tranche d'âge
        print("\nAnalyse des couleurs par tranche d'âge:")
        crosstab = analyze_colors_by_age(df)
        if crosstab is not None:
            print(crosstab)
        
        # Visualiser l'évolution continue des préférences de couleur avec l'âge
        print("\nÉvolution continue des préférences de couleur selon l'âge:")
        color_evolution = plot_color_preferences_by_age_continuous(df)
        print(color_evolution)
        
        # Effectuer l'analyse de régression
        print("\nAnalyse de régression logistique:")
        regression_results = perform_regression_analysis(df)
        print("\nRésumé des résultats de régression:")
        print(regression_results)
        
        print("\nInterprétation des résultats:")
        for color, row in regression_results.iterrows():
            if row['significant']:
                direction = "augmente" if row['odds_ratio'] > 1 else "diminue"
                print(f"- La probabilité de choisir la couleur {color} {direction} significativement avec l'âge.")
                print(f"  Pour chaque année supplémentaire, les odds sont multipliées par {row['odds_ratio']:.3f}.")
            else:
                print(f"- Pas d'effet significatif de l'âge sur le choix de la couleur {color}.")

if __name__ == "__main__":
    main()