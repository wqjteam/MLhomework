from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer


def dictvec():
    dict=DictVectorizer(sparse= False )
    data = dict.fit_transform([{'city':'北京',"temp":100},{'city':'上海',"temp":70},{'city':'深圳',"temp":30}])
    print(dict.get_feature_names())
    print(data)


if __name__ == '__main__':
    dictvec()