?($	??????|j?{?'??w??/???!?? ???$	B?`??@?v?=@?5+G??!D'??d@'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_vO??46<???A$???????Yd?]K???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj??5^?I??A??ǘ????Y??ʡE???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V??-??????A?L?J???Y(~??k	??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a??????%䃞??A?O??e??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}??#J{?/L??Av??????Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C???m4??@??A0*??D??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????o???lV}???A	?c?Z??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???_vO???????AA?c?]K??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-C??6????#?????A???????Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??j+????S??:??A A?c?]??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
1?*?????	?c??A?٬?\m??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?sF????6?;Nё??A?{??Pk??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K???7?A`????A?ڊ?e???Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????C?l????A???_vO??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????Gx$(??Af?c]?F??Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Q????K7?A`??A??y?)??Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0???rh??|??A?j+?????Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6???*??	??Au?V??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???&䃞ͪ??A?\m?????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ???n????A??/?$??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/???f??a?ִ?AL?
F%u??Y?k	??g??*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM?O???!1EqQ?4B@)?W[?????1?g?J?@@:Preprocessing2F
Iterator::ModelU???N@??!?$???%A@)?k	??g??1?l?D"64@:Preprocessing2U
Iterator::Model::ParallelMapV2?HP???!???\g+,@)?HP???1???\g+,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ʡE??!??9?
mP@)?????B??1V??ύ?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??D????!?zi?@)??D????1?zi?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?3??7???!?a]??+@)?St$????1?H???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+???!J?)???2@)?b?=y??1#?7?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*)\???(??!?]???@))\???(??1?]???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9TF??G7@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	(ޚ?N???j?	h???46<???!n????	!       "	!       *	!       2$	[0?A,G??з??L?
F%u??!??ǘ????:	!       B	!       J$	?<?
Q????\?[ޜ???+e?X??!d?]K???R	!       Z$	?<?
Q????\?[ޜ???+e?X??!d?]K???JCPU_ONLYYTF??G7@b Y      Y@q<????C@"?
both?Your program is POTENTIALLY input-bound because 56.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?39.8722% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 