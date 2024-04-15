def linear_regression_1d(data):
    """
    Define a function linear_regression_1d which takes in a list of pairs, where the first value in each pair is the feature value and the second is the response value. Return a pair (m, c) where m is the slope of the line of least squares fit, and c is the intercept of the line of least squares fit. 
    To calculate these values, use the following equations.
    m = (n x.y - ∑x ∑y) / (n x.x - (∑x)2)
    c = (∑y - m∑x)/n
    Here, x is the vector of feature values, and y is the vector of response values, and n is the length of these vectors. 
    """

    #data example
    # [(1,4), (2,7), (3,10)]

    n = len(data)

    # x and y sums
    sum_x = sum(x for x, _ in data)
    sum_y = sum(y for _, y in data)

    # xy and xx sums
    sum_xy = sum(x * y for x, y in data)
    sum_xx = sum(x * x for x, _ in data)

    #slope
    #...the x.y and x.x. are referring to the sum of x.y and sum of x.x.
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)

    #intercept
    c = (sum_y - m * sum_x) / n
    return m, c


#Test
data = [(1, 4), (2, 7), (3, 10)]
m, c = linear_regression_1d(data)
print(m, c)
print(4 * m + c)

# 3.0 1.0
# 13.0