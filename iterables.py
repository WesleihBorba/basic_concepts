# Iterable in big data
sku_list = [7046538, 8289407, 9056375, 2308597]

sku_iterator_object_one = sku_list.__iter__()
sku_iterator_object_two = iter(sku_list)  # Return an object list iterator

for sku in range(0, len(sku_list)):
    next_sku = sku_iterator_object_one.__next__()
    print(next_sku)

