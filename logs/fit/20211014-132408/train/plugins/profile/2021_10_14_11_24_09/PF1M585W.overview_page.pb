?($	?+?ܦ??]?fã7??4??7????!Ǻ?????$	?????T@T??#?@??k$???!?g?5}.)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??QI?????):????Aj?t???Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?߾?3???E??????A??u????Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??K7?A??8gDio??A%u???Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????K??{?/L?
??Aw-!?l??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ?????h??s???A?8??m4??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???3????>W[????A????_v??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I??u????A?5?;N???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4?????ͪ??V??A3ı.n???Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?]K?=??Ϊ??V???A?k	??g??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?镲q???L?J???A????K7??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
6?;Nё???@??ǘ??A??6???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g??^?I+??A'???????YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-????b?=y??AR'??????Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V-????:M??A??D????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L?J??????H??A?镲q??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n??????HP??A????(??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0??
h"lxz??A=?U?????Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ????x?&1??AǺ????Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G?????Q???Aŏ1w-!??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+???9#J{???Am???{???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7????????x???A??(???Y??y?):??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,e?X??!
̚?pB@)?w??#???1??B˖A@:Preprocessing2F
Iterator::Model?????!???\A@)????????1A???)5@:Preprocessing2U
Iterator::Model::ParallelMapV2??ܵ?!?m?)@)??ܵ?1?m?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipi o????!?q??wP@)??d?`T??1???b?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?-????!???-?5@)?-????1???-?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?G?z???!
??f?,@)A??ǘ???1y?2?R?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapTt$?????!Tq?i 2@)??q????1>5?
?U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?l??????!;???w@)?l??????1;???w@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???ڕB@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ńՙ????7???~??????x???!h??s???	!       "	!       *	!       2$	㩲????bk^???????(???!'???????:	!       B	!       J$	??"w?W?????x?y?&1???!?&S???R	!       Z$	??"w?W?????x?y?&1???!?&S???JCPU_ONLYY???ڕB@b Y      Y@q????\V@"?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?89.4512% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 