?($	?0t????????????~?:p???!?ׁsF???$	{>g???@g???@"??1q7@!?6j??6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$? ?	???z6?>W[??Aa2U0*???Y
h"lxz??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5????~j?t???A?K7?A`??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d]?Fx??@?߾???A???S???Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<?R?!???:??H???A?Zd;???Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????C?i?q???A=,Ԛ???Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{????/?'??A?1w-!??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??:?????K7???A?s?????YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^???<,Ԛ???AX?2ı.??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\???㥛? ???A|??Pk???Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???镲??K?46??A?d?`TR??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
/?$????uq???A?ʡE????Yj?q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF????y?):???AX9??v??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s????;Nё\??AԚ?????YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%??C?????????AtF??_??Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ?????????A????????Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx??9??m4???A???(\???Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|гY?????x?&??A?-?????Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????{????A??St$???Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????W?/?'??A???B?i??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h"lxz???㥛? ???AY?8??m??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p????v??/??A?,C????YaTR'????*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU???N@??!=?a???B@)??b?=??1??J6?A@:Preprocessing2F
Iterator::Model??b?=??!??J6?A@)'???????1?B????4@:Preprocessing2U
Iterator::Model::ParallelMapV26?;Nё??!1&??
?,@)6?;Nё??11&??
?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z6?>??!?????3P@)??HP??1WDZ"d?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?'????!?c??(@)??H.???1???@?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceX?5?;N??!Q?F?( @)X?5?;N??1Q?F?( @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapR???Q??!a?y???0@)[B>?٬??1|?O?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*2U0*???!tm??i@)2U0*???1tm??i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??䙨t@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	u:O??B???z????z6?>W[??!???{????	!       "	!       *	!       2$	ꖛkWf???d9q?G???,C????!X9??v??:	!       B	!       J$	U[????NQ?	?????ͪ?Ֆ?!?HP???R	!       Z$	U[????NQ?	?????ͪ?Ֆ?!?HP???JCPU_ONLYY??䙨t@b Y      Y@q?s?<@"?
both?Your program is POTENTIALLY input-bound because 50.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?28.6602% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 