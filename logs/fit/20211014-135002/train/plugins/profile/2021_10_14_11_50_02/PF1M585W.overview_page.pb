?($	?f?)+????(G????ʡE????!io???T??$	5?w@w[??RV??R??`? @!(F.[?&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ʡE?????c?]Kȷ?A$???~???Y?1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????t??????A)\???(??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&RI??&???o???T???A?9#J{???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????B????W?2??A?St$????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&sh??|???M?O????A9??m4???Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ?????A?f???Aq???h ??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI???E???JY??A?????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V?????N@??A???H.??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a?????c?ZB??A
ףp=
??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?=?U????*??	??Aё\?C???Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
5^?I???^)???AX?5?;N??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-??????ͪ????A?A`??"??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?,C???????ׁs??Ao?ŏ1??Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ??w??/???A???h o??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-????????????K??A.???1???Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb???:M???A????Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n??r??????A?? ???Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬???z6?>W??A_?L???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a2?????QI??As??A??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&io???T??aTR'????A>?٬?\??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI???d?]K???A???3???Y??ׁsF??*	fffffR?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9EGr???!V6????A@)?9#J{???1???!?~@@:Preprocessing2F
Iterator::Model0?'???!x??@@)?Q?????18t??[?2@:Preprocessing2U
Iterator::Model::ParallelMapV2???????!?S????*@)???????1?S????*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?H?}8??!?-?{N(@)?H?}8??1?-?{N(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?8??m4??!t??9??P@)Q?|a2??1;E??0&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?S㥛???!Yᠢ?1@)???3???1Vq(???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?t???!???v?	5@)9??v????1xy]???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ZӼ???!2?5|-?@)??ZӼ???12?5|-?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?E]V@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	T?8/?????OzH=????c?]Kȷ?!?:M???	!       "	!       *	!       2$	^X????'?O詽????3???!>?٬?\??:	!       B	!       J$	???V Θ?VT?{?K?=?U??!???Mb??R	!       Z$	???V Θ?VT?{?K?=?U??!???Mb??JCPU_ONLYY?E]V@b Y      Y@q\o????U@"?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?86.8588% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 