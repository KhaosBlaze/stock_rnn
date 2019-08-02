from subconscious import get_X_Y, build_Stanley, get_a_symbol
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from configparser import ConfigParser


varz = ConfigParser()
varz.read('stanley.rekt')

days_to_train_on = varz['BRAINZ']['dtt']

parameters = {'batch_size': list(map(int, varz['BRAINZ']['batch_size'].split(','))),
              'epochs': list(map(int, varz['BRAINZ']['epochs'].split(','))),
              'hu': list(map(int, varz['BRAINZ']['hidden_layer_nodes'].split(','))),
              'ha': list(varz['BRAINZ']['hidden_layer_activation'].split(',')),
              'oa': list(varz['BRAINZ']['output_activation'].split(',')),
              'op': list(varz['BRAINZ']['optimizer'].split(',')),
              'loss': list(varz['BRAINZ']['loss'].split(','))
              }

stanley = KerasClassifier(build_fn = build_Stanley)

grid = GridSearchCV(estimator = stanley,
                    param_grid = parameters,
                    scoring = 'accuracy',
                    cv = 3)

X_train, y_train = get_X_Y(get_a_symbol(), days_to_train_on)

grid.fit(X_train, y_train)

print("The best params are: " + grid.best_params_)
print("HIghest achieved accuracy is: " + grid.best_score_)
