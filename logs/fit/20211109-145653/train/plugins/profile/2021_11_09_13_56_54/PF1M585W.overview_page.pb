?($	}!?*#??#
??0???q?????!X?2ı.??$	??7?@>???.@????`'@!??xJ??+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?	h"lx??X?2ı.??A[B>?٬??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.????u????A8??d?`??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+?????9#J??AM?O????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????F????x??A??Pk?w??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)???ё\?C???A?^)????Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S?????:pΈ??A&S??:??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u???????镲??A???z6??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+??E???JY??A?A?f????YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{?????1??%??A?s????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?_?L??RI??&???A???????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
Ӽ????y?&1???A?Q?????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????߾????j+????AV-???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?,C??????4?8E??A#??~j???Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/???):????A?W?2??Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_??~8gDi??AM?J???Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\???$???????Aw-!?l??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w??5?8EGr??Ag??j+???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??~j?t???HP???A?X????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????ׁsF???A?9#J{???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U???N@??m???{???A46<???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q?????#??~j???A???Mb??Y?k	??g??*	23333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-??!?^}I?oA@)L?
F%u??1tU?d>@:Preprocessing2F
Iterator::Model@?߾???!?aI`%@@)x??#????1+uyʿ3@:Preprocessing2U
Iterator::Model::ParallelMapV2X9??v??!?0;??)@)X9??v??1?0;??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???(\???!!O?Op?P@)V????_??1#Z?G?k"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?	???!??,5}/@)??u????1פ??X!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???镲??!8Ґ???7@)???&??1?p????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceŏ1w-??!??~?I@)ŏ1w-??1??~?I@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??(\?¥?!?"??\?@)??(\?¥?1?"??\?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??'??}@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???????+?V??X?2ı.??!m???{???	!       "	!       *	!       2$	+??D?????V?ռ?M?J???!8??d?`??:	!       B	!       J$	8??v?????R?Z}7???]K?=??!?St$????R	!       Z$	8??v?????R?Z}7???]K?=??!?St$????JCPU_ONLYY??'??}@b Y      Y@q??gP?@@"?
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?33.4243% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 