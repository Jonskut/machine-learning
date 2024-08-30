
def my_sort(my_list):

    # using ready functions
    sorted_list1 = my_list
    sorted_list1.sort()

    # using "own" function
    sorted_list2 = my_list

    # bubble sort
    for i in range(len(sorted_list2)):
        is_sorted = True

        for j in range(len(sorted_list2) - i - 1):
            if sorted_list2[j] > sorted_list2[j + 1]:
                sorted_list2[j + 1], sorted_list2[j] = \
                    sorted_list2[j], sorted_list2[j + 1]

                is_sorted = False

        if is_sorted:
            break

    return sorted_list1, sorted_list2


if __name__ == '__main__':

    sorted_list1, sorted_list2 = my_sort(my_list)
