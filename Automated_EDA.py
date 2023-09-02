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
		 
	elif file_path.endswith("csv"):
		data=pd.read_csv(file_path)
		print(data.head(10))
		
	else:
		raise valueError("File Format is not supported")
	#clean_Missing_data	
	file=data.dropna()
	files=file.reset_index()
		
	# draw_plot_of_each_numerical_column
	numeric_list=files.select_dtypes(include=['number']).columns
	for i in numeric_list:  
		plt.plot(files[i])
		plt.xlabel(i)
		plt.ylabel("number")
		plt.title(i)
		plt.show()
	#draw_bar_of_each_column	
	columns_list=[]
	# print("iam here")
	columns_list=files.select_dtypes(include=['number']).columns
	for i in columns_list:  
		files[i].plot(kind="bar",xlabel=i,ylabel="number",title=i)
		plt.show()
	#draw_heat_map
	sns.heatmap(files.corr())
	plt.show()
	#get_duplicated_columns
	print(files.duplicated())
	#drop_duplciate_columns
	files=files.drop_duplicates(files)
	print(files)
	#show_types_of_columns
	print(files.dtypes)
	#draw_scatter_plot
	numeric_list=files.select_dtypes(include=['number']).columns
	for i in numeric_list:
		column_data = files[i]
		x_values = range(len(column_data))
		plt.scatter(x_values, column_data)
		plt.xlabel(i)
		plt.ylabel("Values")
		plt.title(i)
		plt.show()

	#dimensionality_reduction
	n_components=2
	pca = PCA(n_components=n_components)
	reduced_df = pca.fit_transform(files[numeric_list])

  	# Create a new dataset with the reduced dimensionality.
	reduced_df = pd.DataFrame(reduced_df, columns=[f"PC{i}" for i in range(n_components)])

	print(reduced_df)


get_data_from_file("D:\\Mucis recommetion\\Music_recommendation\\members.csv")





# def feature_selection(datat, target_column, n_features=10):
 
#   # Select the features with the highest correlation with the target variable.
#   selector = SelectKBest(f_classif, k=n_features)
#   selected_features = selector.fit_transform(datat.drop(target_column, axis=1), datat[target_column])

#   # Create a new dataset with the selected features.
#   selected_df = pd.DataFrame(selected_features, columns=datat.drop(target_column, axis=1).columns[selector.get_support()])

#   return selected_df


