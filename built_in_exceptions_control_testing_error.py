# DEFINE THE GOALS MAYBE CLUSTER
import unittest

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

