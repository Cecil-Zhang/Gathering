---
title:  "R cheatsheet"
date:   2022-04-22 12:00:04 -0700
categories: 
 - Programming Language
toc: true
---

## Data Types
### Basic Data Types
- Numeric (Float/Double): 10.5, 5e-06
- Integer: 3, -3, 7L
- Complex: 2+3i, complex(real=2, imaginary=3)
- Logical: TRUE, FALSE, T, F
- Character: "a", "2.718"

### Vector
- A **vector** is a sequence of elements of the same basic data type
- Construction: using c(), :, rep(), seq(), output of a function

```r
x <- c(1,2,3,5.5), x <- c("a", "b", "abc"), x <- c(T, F, F, T)
y <- 10:15 # unit spacing
y <- rep(2,6) # 2 2 2 2 2 2
z <- seq(0,1,by=0.2) # spacing by 0.2, 0.0 0.2 0.4 0.6 0.8 1.0
v <- sample(1:5, size=10, replace=TRUE)
```

- Construct named vector: v <- c(x=2, y=3, z=10.2, ...)
- Assign names to vector: names(v) <- c(‘x’, ‘y’, ‘z’, ‘a’, ‘b’, ...)
- Indexing (index starts with 1 instead of 0)

### List
- A **list** is a sequence of objects of different data types and lengths.
- Construction: using list()
    - L <- list(5, c(1,2,3), matrix(rep(2,6),2,3), list(5, "abc"))
    - L becomes a sequence of a number, vector, matrix and list
- Naming (similar to vectors)
    - names(L) <- c("num", "vec", "mat", "list")
- Indexing:
    - L[1] # 5
    - L["num"] or L$num # 5

### Data Frame
- A **data frame** is a list of vectors (of different data types) of equal
length.
- It is used for storing data table. The vectors are the columns of the
table.
- Construction: data.frame(), read.table(), read.xls(), read.csv()


## Data Frame Operations

### Naming
```r
names(df) <- c(‘x’, ‘y’, ‘z’, ‘a’, ‘b’, ...) 
names(v)[4] <- "tom"
```

### Indexing
```r
# Single Indexing
df[2,3] # get the element located at (row=2,col=3)
df["person1", "age"] # get the age column of the person1 row

# Column Indexing
df[1:10] # get the first 10 columns
df[,2] # get the second column
df["age"] # get the age column
df$age # get the age column

# Segement Indexing
df[1:10,] # get the first 10 rows with all columns
df[c(r1, r2, ..), c(c1, c2, ..)] # gives the sub-matrix/sub-dataframe of selected rows and columns
df[c(T, F, T, ..), c(F, F, T, ..)] # gives the sub-matrix /sub-dataframe for selected rows and columns corresponding to the "T" values
df[df[,2]<40, ] # gives the sub-matrix /sub-dataframe with those rows whose 2nd column is less than 40
df$age[df$age<40] <- 30 # setting with selective indexing

# negative index
df[,-1] # get all columns except the first column
```


## Loop and Condition
```r
for (i in 1:5){
	print(i)
}
# 1,2,3,4,5

if(condition){ 
	statement
} else if (condition2) { 
	statement2
} else { 
	statement3
}
```

## Function

```r
testFunc <- function(x) print(x)
testFunc(2)
# 2

function_name <- function(arguments){ 
	function_definition
	return (output) # or just write: output, last computed variable is returned 
}
```

## Useful functions

- length: get the length of object
- head(obj, n=10): get the first 10 elements of obj
- tail(obj, n=10): get the last 10 elements of obj
- tapply(df$age, df$gender, mean): get the mean age of each gender, like groupBy in sql
- split(df$age, df$gender): split the age data into groups of gender