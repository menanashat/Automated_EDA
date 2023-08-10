import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
def get_data_from_file(file_path):
	if file_path.endswith("xls") or file_path.endswith("xlsx"):
		data=pd.read_excel(file_path)
		print(data.head(10))	
		return data 
	elif file_path.endswith("csv"):
		data=pd.read_csv(file_path)
		print(data.head(10))
		return data
	else:
		raise valueError("File Format is not supported")

def clean_Missing_data(data):
	file=data.dropna()
	files=file.reset_index()
	return files

def draw_plot_of_each_numerical_column(data):
	numeric_list=data.select_dtypes(include=['number']).columns
	for i in numeric_list:  
		plt.plot(data[i])
		plt.xlabel(i)
		plt.ylabel("number")
		plt.title(i)
		plt.show()

def draw_bar_of_each_column(data):
	columns_list=[]
	columns_list=data.columns
	for i in columns_list:  
		data[i].plot(kind="bar",xlabel=i,ylabel="number",title=i)
		plt.show()

def draw_scatter_plot(data, column_name):
    column_data = data[column_name]
    x_values = range(len(column_data))
    plt.scatter(x_values, column_data)
    plt.xlabel(column_name)
    plt.ylabel("Values")
    plt.title(column_name)
    plt.show()

def draw_heat_map(datas):
	sns.heatmap(datas.corr())
	plt.show()

def get_duplicated_columns(data):
	print(data.duplicated())

def drop_duplciate_columns(data,columns_name):
	return data.drop_duplicates(columns_name
		)

def show_types_od_columns(data):
	print(data.dtypes)

def feature_selection(datat, target_column, n_features=10):
 
  # Select the features with the highest correlation with the target variable.
  selector = SelectKBest(f_classif, k=n_features)
  selected_features = selector.fit_transform(datat.drop(target_column, axis=1), datat[target_column])

  # Create a new dataset with the selected features.
  selected_df = pd.DataFrame(selected_features, columns=datat.drop(target_column, axis=1).columns[selector.get_support()])

  return selected_df

def dimensionality_reduction(data, n_components=2):

  # Perform PCA on the dataset.
  pca = PCA(n_components=n_components)
  reduced_df = pca.fit_transform(data)

  # Create a new dataset with the reduced dimensionality.
  reduced_df = pd.DataFrame(reduced_df, columns=[f"PC{i}" for i in range(n_components)])

  return reduced_df	


data=get_data_from_file("D:\\New folder (4)\\creditcard.csv")

clean_Data=clean_Missing_data(data)

# draw_bar_of_each_column(clean_Data) 

# draw_scatter_plot(clean_Data,"fg_apt_id")
# draw_hear_map(clean_Data)
# get_duplicated_columns(clean_Data)

# print(clean_Data['fg_apt'].unique())
# show_types_od_columns(clean_Data)
# clean_Data_without_duplicate=drop_duplciate(clean_Data,'type')
# print(clean_Data_without_duplicate)

asd=dimensionality_reduction(clean_Data)
print(asd)