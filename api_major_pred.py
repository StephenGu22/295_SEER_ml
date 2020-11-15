import numpy as np
from sklearn.metrics import pairwise_distances

def major_recommendation(score_list):
    name_list = [['Construction Trades', 'Family and Consumer Sciences/Human Sciences ',
                  'Mechanic and Repair Technologies/Technician', 'Parks, Recreation, Leisure and Fitness Studies',
                  'Personal and Culinary Services', 'Public Administration and Social Services Professions',
                  'Security and Protective Services', 'Transportation and Materials Moving'],
                 ['Area, Ethnic, Cultural and Gender Studies', 'Communication, Journalism and Related Programs',
                  'Foreign Languages, Literature, Linguistics', 'History', 'Legal Professions and Studies',
                  'Theology and Religious Vocations'],
                 ['Biological and Bio-medical Sciences', 'English Language and Literature/Letters',
                  'Liberal Arts and Sciences, General Studies, and Humanities', 'Library Science And Administration',
                  'Natural Resources and Conservation', 'Philosophy and Religious Studies', 'Social Science'],
                 ['Agriculture', 'Architecture Related Services',
                  'Business Management, Marketing, and Related Support Services', 'Education',
                  'Engineering Technologies/Technicians', 'Health Professions and Related Clinical Services',
                  'Military Technologies And Applied Sciences', 'Psychology', 'Visual and Performing Arts'],
                 ['Computer and Information Sciences and Support Services', 'Engineering',
                  'Mathematics and Statistics ', 'Multi/Interdisciplinary Studies', 'Physical Sciences']]
    centers = [[521.83333333, 502.5, 512.83333333],
               [548.57142857, 523.42857143, 531.85714286],
               [485.66666667, 499, 482.88888889],
               [541.6, 579.2, 535.4],
               [438.75, 449.25, 435.75]]
    student_score = np.array(score_list[:3])
    min_index = 0
    min_value = pairwise_distances([student_score], [centers[0]])[0][0]
    for i in range(5):
        curr = pairwise_distances([student_score], [centers[i]])[0][0]
        if min_value > curr:
            min_value = curr
            min_index = i
    return name_list[min_index]

