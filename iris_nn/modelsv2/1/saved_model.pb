ŞŢ
Đ§
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
Ą
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
b
InitializeTableV2
table_handle
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
,
Log
x"T
y"T"
Ttype:

2
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ď
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.8.02v1.8.0-0-g93bc2e20728őľ
@

tf_examplePlaceholder*
dtype0*
_output_shapes
:
U
ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
b
ParseExample/ParseExample/namesConst*
valueB *
dtype0*
_output_shapes
: 
m
&ParseExample/ParseExample/dense_keys_0Const*
valueB BX_data*
dtype0*
_output_shapes
: 

ParseExample/ParseExampleParseExample
tf_exampleParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0ParseExample/Const*
Tdense
2*
Ndense*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Nsparse *
sparse_types
 *
dense_shapes
:
_
X_dataIdentityParseExample/ParseExample*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
random_normal/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*

seede*
T0*
dtype0*
seed2
*
_output_shapes

:
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes

:*
T0
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:
X
Variable
VariableV2*
shape
:*
dtype0*
_output_shapes

:
x
Variable/AssignAssignVariablerandom_normal*
T0*
_class
loc:@Variable*
_output_shapes

:
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:
f
random_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_1/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
˘
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
seed2*
_output_shapes

:*

seede*
T0

random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes

:
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes

:*
T0
Z

Variable_1
VariableV2*
dtype0*
_output_shapes

:*
shape
:

Variable_1/AssignAssign
Variable_1random_normal_1*
_class
loc:@Variable_1*
_output_shapes

:*
T0
o
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:
_
random_normal_2/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_2/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
T0*
dtype0*
seed2*
_output_shapes
:*

seede
}
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*
_output_shapes
:
f
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*
_output_shapes
:
R

Variable_2
VariableV2*
dtype0*
_output_shapes
:*
shape:
|
Variable_2/AssignAssign
Variable_2random_normal_2*
_output_shapes
:*
T0*
_class
loc:@Variable_2
k
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes
:
_
random_normal_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_3/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
T0*
dtype0*
seed2%*
_output_shapes
:*

seede
}
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes
:*
T0
f
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*
_output_shapes
:
R

Variable_3
VariableV2*
dtype0*
_output_shapes
:*
shape:
|
Variable_3/AssignAssign
Variable_3random_normal_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:*
T0*
_class
loc:@Variable_3
Y
MatMulMatMulX_dataVariable/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
AddAddMatMulVariable_2/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
ReluReluAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
MatMul_1MatMulReluVariable_1/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
Add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
SoftmaxSoftmaxAdd_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
LogLogSoftmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
mulMulPlaceholderLog*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 
K
SumSummulSum/reduction_indices*
_output_shapes
:*
T0
4
NegNegSum*
_output_shapes
:*
T0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
9
MeanMeanNegConst*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
]
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
~
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
_output_shapes
:
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB:*
dtype0
}
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*
T0*
_output_shapes
:
`
gradients/Mean_grad/Const_1Const*
valueB
 *  @@*
dtype0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes
:
_
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
_output_shapes
:*
T0
K
gradients/Sum_grad/ShapeShapemul*
T0*
_output_shapes
:

gradients/Sum_grad/SizeConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
Ą
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 

gradients/Sum_grad/Shape_1Const*
valueB *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/startConst*
value	B : *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/deltaConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ă
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:

gradients/Sum_grad/Fill/valueConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
¨
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ú
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Sum_grad/Maximum/yConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ŕ
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/DynamicStitch*
T0*
_output_shapes
:

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
S
gradients/mul_grad/ShapeShapePlaceholder*
T0*
_output_shapes
:
M
gradients/mul_grad/Shape_1ShapeLog*
T0*
_output_shapes
:
Ť
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
m
gradients/mul_grad/MulMulgradients/Sum_grad/TileLog*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
gradients/mul_grad/Mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Ú
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1

gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
$gradients/Softmax_grad/Reshape/shapeConst*
valueB"˙˙˙˙   *
dtype0*
_output_shapes
:

gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ą
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Add_1_grad/SumSumgradients/Softmax_grad/mul_1*gradients/Add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Add_1_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes
:*
T0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
â
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ű
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
_output_shapes
:
­
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_1/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(

 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ě
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
é
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:

gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
gradients/Add_grad/ShapeShapeMatMul*
T0*
_output_shapes
:
d
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ť
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes
:*
T0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
Ú
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
_output_shapes
:
§
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0

gradients/MatMul_grad/MatMul_1MatMulX_data+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
b
GradientDescent/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
í
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable*
_output_shapes

:
ő
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:
ě
6GradientDescent/update_Variable_2/ApplyGradientDescentApplyGradientDescent
Variable_2GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
_output_shapes
:*
T0
î
6GradientDescent/update_Variable_3/ApplyGradientDescentApplyGradientDescent
Variable_3GradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:
ů
GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent7^GradientDescent/update_Variable_2/ApplyGradientDescent7^GradientDescent/update_Variable_3/ApplyGradientDescent

Const_1Const*Y
valuePBNBSpecies_Iris-setosaBSpecies_Iris-versicolorBSpecies_Iris-virginica*
dtype0*
_output_shapes
:
V
index_to_string/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
]
index_to_string/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
]
index_to_string/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

index_to_string/rangeRangeindex_to_string/range/startindex_to_string/Sizeindex_to_string/range/delta*
_output_shapes
:
j
index_to_string/ToInt64Castindex_to_string/range*

SrcT0*

DstT0	*
_output_shapes
:
Z
index_to_stringHashTableV2*
value_dtype0*
_output_shapes
: *
	key_dtype0	
]
index_to_string/ConstConst*
valueB BUNKNOWN*
dtype0*
_output_shapes
: 
z
index_to_string/table_initInitializeTableV2index_to_stringindex_to_string/ToInt64Const_1*

Tkey0	*

Tval0
4
init_all_tablesNoOp^index_to_string/table_init
J
TopKV2/kConst*
_output_shapes
: *
value	B :*
dtype0
h
TopKV2TopKV2SoftmaxTopKV2/k*
T0*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
W
CastCastTopKV2:1*

SrcT0*

DstT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

index_to_string_LookupLookupTableFindV2index_to_stringCastindex_to_string/Const*	
Tin0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tout0
Z
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign
6
init_all_tables_1NoOp^index_to_string/table_init
\
init_1NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign

init_2NoOp
6
init_all_tables_2NoOp^index_to_string/table_init
8

group_depsNoOp^init_1^init_2^init_all_tables_2
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_c4dafe6f190f4ca8b062872a95415053/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/SaveV2/tensor_namesConst"/device:CPU:0*A
value8B6BVariableB
Variable_1B
Variable_2B
Variable_3*
dtype0*
_output_shapes
:
z
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
´
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2
Variable_3"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst"/device:CPU:0*A
value8B6BVariableB
Variable_1B
Variable_2B
Variable_3*
dtype0*
_output_shapes
:
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
Ž
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
u
save/AssignAssignVariablesave/RestoreV2*
_class
loc:@Variable*
_output_shapes

:*
T0
}
save/Assign_1Assign
Variable_1save/RestoreV2:1*
T0*
_class
loc:@Variable_1*
_output_shapes

:
y
save/Assign_2Assign
Variable_2save/RestoreV2:2*
T0*
_class
loc:@Variable_2*
_output_shapes
:
y
save/Assign_3Assign
Variable_3save/RestoreV2:3*
T0*
_class
loc:@Variable_3*
_output_shapes
:
X
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"3
table_initializer

index_to_string/table_init"ˇ
trainable_variables
?

Variable:0Variable/AssignVariable/read:02random_normal:0
G
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:0
G
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:0
G
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:0"
train_op

GradientDescent",
saved_model_main_op

init_all_tables_1"­
	variables
?

Variable:0Variable/AssignVariable/read:02random_normal:0
G
Variable_1:0Variable_1/AssignVariable_1/read:02random_normal_1:0
G
Variable_2:0Variable_2/AssignVariable_2/read:02random_normal_2:0
G
Variable_3:0Variable_3/AssignVariable_3/read:02random_normal_3:0*ľ
predict_iris¤

inputs
tf_example:0:
classes/
index_to_string_Lookup:0˙˙˙˙˙˙˙˙˙)
scores
TopKV2:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*x
predict_clickg

inputs
tf_example:0)
scores
TopKV2:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict