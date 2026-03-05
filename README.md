# README.md
This CSV file contains the complete dataset of 5,000 university students. It includes demographic details, academic history, and lifestyle factors. The target variable is Final_CGPA, making this file ready for regression analysis and grade prediction models. The data is pre-cleaned with no missing values.

Source : https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset

A l'aide de  : nombre d'heures de sommeil, le précédent gpa (grade point average), l'âge, le genre, la présence (%) ... nous allons chercher à prédire le final gpa.

La tâche est donc une régression, la taille du fichier est de 274 kb, composé de 5000 étudiants avec 8 paramètres (features) avec 2 catégories.
Pas de valeur manquante.



Column Name 	Description 	Type
Student_ID 	Unique identifier for each student ( ID00001) 	String
Gender 	Gender of the student (Male/Female) 	Categorical
Age 	Age of the student (18–24 years) 	Integer
Major 	Field of study ( Computer Science, Engineering) 	Categorical
Attendance_Pct 	Percentage of classes attended (0–100%) 	Float
Study_Hours_Per_Day 	Average hours spent studying per day 	Float
Previous_CGPA 	GPA from the previous semester (0.0–4.0 scale) 	Float
Sleep_Hours 	Average hours of sleep per night 	Float
Social_Hours_Week 	Average hours spent socializing/partying per week 	Integer
Final_CGPA 	Target Variable: final cumulative GPA (0.0–4.0 scale) 	Float

License 
CC BY-NC-SA 4.0

