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

# Configuration des graphiques
try:
    plt.style.use('seaborn')
except:
    pass  # Utiliser le style par défaut si seaborn n'est pas disponible
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
        'NOIR': r'noir|black|anthracite|carbone|graphite|Saphirschwarz',
        'BLANC': r'blanc|white|nacré|perle|ivory|Bianco|Alpinweiss|Ivoire',
        'GRIS': r'gris|grey|gray|metal|argent|silver|Acier Moderne|Grau|mondsteingrau|Grigio',
        'BLEU': r'bleu|blue|azur|marine|turquoise|cyan|Blu|Mediterranblau',
        'ROUGE': r'rouge|red|bordeaux|carmin|rubis|burgundy',
        'VERT': r'vert|green|emeraude|olive|kaki',
        'JAUNE': r'jaune|yellow|or|gold|citron|Sable (M0EU)|Sable',
        'MARRON': r'marron|brown|chocolat|beige|camel|taupe|bronze|Brun Terracotta (CNZ)|Brun Terracotta (CNZ)',
        'ORANGE': r'orange|amber|cuivre|copper',
        'VIOLET': r'violet|purple|mauve|lilas|lavande',
        'ROSE': r'rose|pink|fuchsia|magenta'
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

def analyze_colors_by_age(df):
    """Analyser les couleurs préférées par tranche d'âge"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_commandes = df[df['FLAG_COMMANDE'] == 1].copy()
    
    # Vérifier que les colonnes nécessaires existent
    if 'COULEUR_STANDARD' not in df_commandes.columns or 'tranche_age' not in df_commandes.columns:
        print("Colonnes requises manquantes.")
        return None
    
    # Enlever les valeurs manquantes
    df_commandes = df_commandes.dropna(subset=['COULEUR_STANDARD', 'tranche_age'])
    
    # 1. Tableau croisé des couleurs par tranche d'âge
    crosstab = pd.crosstab(df_commandes['tranche_age'], df_commandes['COULEUR_STANDARD'])
    
    # Normaliser par ligne pour avoir des pourcentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # 2. Visualiser avec une heatmap
    plt.figure(figsize=(16, 10))
    
    # Utiliser toutes les couleurs standardisées, elles sont déjà regroupées
    crosstab_pct_filtered = crosstab_pct
    
    sns.heatmap(crosstab_pct_filtered, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Pourcentage de chaque couleur de véhicule par tranche d\'âge', fontsize=16)
    plt.xlabel('Couleur du véhicule', fontsize=14)
    plt.ylabel('Tranche d\'âge', fontsize=14)
    plt.tight_layout()
    plt.savefig('heatmap_couleur_age.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Barplot pour chaque tranche d'âge
    plt.figure(figsize=(18, 12))
    
    # Créer un subplot pour chaque tranche d'âge
    age_groups = sorted(df_commandes['tranche_age'].unique())
    n_groups = len(age_groups)
    rows = (n_groups + 1) // 2  # Arrondir au supérieur
    
    for i, age_group in enumerate(age_groups):
        plt.subplot(rows, 2, i+1)
        
        # Sélectionner les données pour cette tranche d'âge
        age_data = df_commandes[df_commandes['tranche_age'] == age_group]
        
        # Compter les occurrences de chaque couleur standardisée
        color_counts = age_data['COULEUR_STANDARD'].value_counts()
        
        # Créer le graphique
        sns.barplot(x=color_counts.index, y=color_counts.values)
        plt.title(f'Répartition des couleurs pour la tranche d\'âge {age_group}')
        plt.xlabel('Couleur')
        plt.ylabel('Nombre de véhicules')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('barplot_couleur_par_age.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return crosstab_pct

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
    
    # Visualiser les coefficients de régression
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df.index, results_df['coefficient'])
    
    # Colorer selon la significativité
    for i, bar in enumerate(bars):
        if results_df.iloc[i]['significant']:
            bar.set_color('darkblue')
        else:
            bar.set_color('lightblue')
            
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Coefficient de l\'âge sur la probabilité de choisir chaque couleur')
    plt.xlabel('Couleur')
    plt.ylabel('Coefficient (log-odds)')
    plt.xticks(rotation=45, ha='right')
    
    # Ajouter une note sur la significativité
    plt.figtext(0.5, 0.01, 'Les barres foncées indiquent des coefficients statistiquement significatifs (p<0.05)',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('regression_age_couleur.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualiser les odds ratios
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df.index, results_df['odds_ratio'])
    
    # Colorer selon la significativité
    for i, bar in enumerate(bars):
        if results_df.iloc[i]['significant']:
            bar.set_color('darkgreen')
        else:
            bar.set_color('lightgreen')
            
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    plt.title('Odds Ratio de l\'âge sur la probabilité de choisir chaque couleur')
    plt.xlabel('Couleur')
    plt.ylabel('Odds Ratio')
    plt.xticks(rotation=45, ha='right')
    
    # Ajouter une note explicative
    plt.figtext(0.5, 0.01, 'OR > 1: La probabilité augmente avec l\'âge, OR < 1: La probabilité diminue avec l\'âge',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('odds_ratio_age_couleur.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def plot_color_preferences_by_age_continuous(df):
    """Créer un graphique montrant l'évolution des préférences de couleurs avec l'âge (continu)"""
    # Filtrer pour n'avoir que les lignes avec commande
    df_commandes = df[df['FLAG_COMMANDE'] == 1].copy()
    
    # Enlever les valeurs manquantes
    df_commandes = df_commandes.dropna(subset=['COULEUR_STANDARD', 'age_client'])
    
    # Utiliser toutes les couleurs standardisées
    df_filtered = df_commandes.copy()
    
    # Créer des bins d'âge plus fins pour une visualisation continue
    age_bins = np.arange(20, 80, 5)  # tranches d'âge de 5 ans
    df_filtered['age_bin'] = pd.cut(df_filtered['age_client'], bins=age_bins)
    
    # Calculer la distribution des couleurs pour chaque bin d'âge
    color_distribution = pd.crosstab(df_filtered['age_bin'], df_filtered['COULEUR_STANDARD'], normalize='index') * 100
    
    # Convertir les bins en valeurs numériques pour le graphique (point médian de chaque bin)
    age_mid_points = [(b.left + b.right) / 2 for b in color_distribution.index]
    
    # Créer le graphique
    plt.figure(figsize=(14, 8))
    
    # Tracer une ligne pour chaque couleur standardisée
    for color in color_distribution.columns:
        plt.plot(age_mid_points, color_distribution[color], marker='o', linewidth=2, label=color)
    
    plt.title('Évolution des préférences de couleur selon l\'âge', fontsize=16)
    plt.xlabel('Âge', fontsize=14)
    plt.ylabel('Pourcentage (%)', fontsize=14)
    plt.legend(title='Couleur')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('evolution_couleur_age_continu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return color_distribution

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