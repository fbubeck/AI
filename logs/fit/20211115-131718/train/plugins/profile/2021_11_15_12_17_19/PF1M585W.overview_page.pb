?($	?y3}????|??z????4?8EG??!??????$	y??,@A3
j@M??@!??.C??2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ϊ??V????9#J{???A:??H???YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8????46<?R??A}гY????Y:??H???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/???ZB>????A?/L?
F??Y?-?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O?????Ǻ????A#J{?/L??Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4??ffffff??AΈ?????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F??      ??A?G?z??Y?H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????ׁsF???A?!?uq??Yx$(~???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????sh??|???ATR'?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H.??.?!??u??A???K7???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?鷯????m4????A(??y??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
s??A??Z??ڊ???A?%䃞???Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S????C??????AV-????Y??#?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?z?G??NbX9???AV-????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?|??u????A?"??~j??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m??????????Ag??j+???Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????2??%????AI??&??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Dio?????)\???(??A~8gDi??YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c??????????Aףp=
???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??<,Ԛ????HP??A-C??6??Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U???????6???A??????Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG??e?`TR'??AF%u???Y?sF????*	333331?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV????_??!D'?t?@@)'1?Z??1>?EhO?>@:Preprocessing2F
Iterator::ModelӼ????!?e9Z?f@@)g??j+???1?Up3T4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???B?i??!M?R??P@)?Q???1???w?)@:Preprocessing2U
Iterator::Model::ParallelMapV2@?߾???!??؈??(@)@?߾???1??؈??(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?O??e??!??E7??4@)??&???1???h"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?ڊ?e???!?:ǫ*@)?K7?A`??1? lt*1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?i?q????!U???c&@)?i?q????1U???c&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*vq?-??!?H?ϔ@)vq?-??1?H?ϔ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?[?"?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?^?]????ҧHhE???e?`TR'??!???????	!       "	!       *	!       2$	?~?T?
??h?˹????F%u???!V-????:	!       B	!       J$	??ʡE??????܉?c?ZB>???!M?O???R	!       Z$	??ʡE??????܉?c?ZB>???!M?O???JCPU_ONLYY?[?"?@b Y      Y@q%!⢐d@@"?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.7857% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 