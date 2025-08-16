# We will provoke errors using Logistic regression
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test data, to test the model
y_prediction = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_prediction)
classification_report = classification_report(y_test,y_prediction)
print(f'accuracy score: {accuracy}')
print(f'Classification report : \n  {classification_report}')

exit()
# Creating a class to build logistic regression
class LogisticRegressionAnalysis:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


# Testing Number of classes

# Raise error:
def open_register(employee_status):
    if employee_status == 'Authorized':
        print('Successfully opened cash register')
    else:
        # Alternatives: raise TypeError() or TypeError('Message')
        raise TypeError


# Try and except
colors = {
    'red': '#FF0000',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
}

for color in ('red', 'green', 'yellow'):
    try:
        print('The hex value of ' + color + ' is ' + colors[color])
    except:
        print('An exception occurred! Color does not exist.')
    print('Loop continues...')


# Handling Multiple Exceptions

instrument_prices = {
  'Banjo': 200,
  'Cello': 1000,
  'Flute': 100,
}

def display_discounted_price(instrument, discount):
  full_price = instrument_prices[instrument]
  discount_percentage = discount / 100
  discounted_price = full_price - (full_price * discount_percentage)
  print("The instrument's discounted price is: " + str(discounted_price))

instrument = 'Banjo'
discount = 20

# Write your code below:
try:
  display_discounted_price(instrumenta, discount)
except KeyError:
    print('An invalid instrument was entered!')
except TypeError:
    print('Discount percentage must be a number!')
except Exception:
    print('Hit an exception other than KeyError or TypeError!')

# ADD THE ELSE:
try:
    pass
    # check_password()
except ValueError:
    print('Wrong Password! Try again!')
else:
    pass
    # login_user()
    # 20 other lines of imaginary code

# ADD THE FINALLY CLAUSE:

try:
    pass
except ValueError:
    print('Wrong Password! Try again!')
else:
    pass
    # 20 other lines of imaginary code
finally:
    pass


# User-defined Exceptions
class LocationTooFarError(Exception):
    pass


# Write your code below (Checkpoint 1 & 2)
class InventoryError(Exception):
    def __init__(self, supply):
        self.supply = supply

    def __str__(self):
        return 'Available supply is only ' + str(self.supply)


inventory = {
    'Piano': 3,
    'Lute': 1,
    'Sitar': 2
}


def submit_order(instrument, quantity):
    supply = inventory[instrument]
    # Write your code below (Checkpoint 3)
    if quantity > supply:
        raise InventoryError(supply)
    else:
        inventory[instrument] -= quantity
        print('Successfully placed order! Remaining supply: ' + str(inventory[instrument]))


instrument = 'Piano'
quantity = 5
submit_order(instrument, quantity)


# UnitTest

# Function that gets tested
def times_ten(number):
    return number * 100


# Test class
class TestTimesTen(unittest.TestCase):
    def test_multiply_ten_by_zero(self):
        self.assertEqual(times_ten(0), 0, 'Expected times_ten(0) to return 0')

    def test_multiply_ten_by_one_million(self):
        self.assertEqual(times_ten(1000000), 10000000, 'Expected times_ten(1000000) to return 10000000')

    def test_multiply_ten_by_negative_number(self):
        self.assertEqual(times_ten(-10), -100, 'Expected add_times_ten(-10) to return -100')


# Run the tests
unittest.main()

# Assert Methods III: Exception and Warning Methods

import warnings

class PowerError(Exception):
    pass

class WaterLevelWarning(Warning):
    pass

def power_outage_detected(outage_detected):
    if outage_detected:
        raise PowerError('A power outage has been detected somewhere in the system')
    else:
        print('All systems receiving power')

def water_levels_check(liters):
    if liters < 200:
        warnings.warn('Water levels have fallen below 200 liters', WaterLevelWarning)
    else:
        print('Water levels are adequate')


import unittest
import alerts


# Write your code here:
class SystemAlertTests(unittest.TestCase):
    def test_power_outage_alert(self):
        self.assertRaises(alerts.PowerError, alerts.power_outage_detected, True)

    def test_water_levels_warning(self):
        self.assertWarns(alerts.WaterLevelWarning, alerts.water_levels_check, 150)


unittest.main()


# Test Fixtures

import unittest
#import kiosk


class CheckInKioskTests(unittest.TestCase):

    def test_check_in_with_flight_number(self):
        print('Testing the check-in process based on flight number')

    def test_check_in_with_passport(self):
        print('Testing the check-in process based on passport')

    # Write your code below:
    @classmethod
    def setUpClass(cls):
        pass
        #kiosk.power_on_kiosk()

    @classmethod
    def tearDownClass(cls):
        pass
        #kiosk.power_off_kiosk()

    def setUp(self):
        pass
        #kiosk.return_to_welcome_page()


unittest.main()

