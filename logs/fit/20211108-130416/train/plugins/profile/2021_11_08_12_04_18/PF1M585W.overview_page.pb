?($	r8
dX??O?թ3???????x???!\ A?c???$	\??y??@? h:?@?k?T??@!??V5/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???????A?c?]K??A0*??D??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]???HP???A?[ A???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F????x??io???T??A+??????Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H???sF????A?+e?X??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??B?i???r??????A6?>W[???Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???V?/??B?f??j??AD?l?????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF??????m4????A??1??%??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ǘ???????ׁsF??A?.n????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+???OjM???A?(??0??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?MbX9??q?-???A&S????YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?sF?????|a2U??Avq?-??Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&p_?Q????ܵ?|??A????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f??????(\????A$???~???Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???4??@????A؁sF????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??? ?r???X????A?sF????Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Tt$?????/?$???A#??~j???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ׁs???8??m4??A?3??7???Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j???bX9????A!?rh????Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&\ A?c?????^??AԚ?????Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l???/?$??A?;Nё\??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????x????a??4???A;M?O??Y46<?R??*	43333o?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??4?8E??!???L0?A@)m???????1#?5v@@:Preprocessing2F
Iterator::Model?(??0??!?	?>???@)??:M???1?،?P?3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?b?=y??!??L0?Q@)R'??????1?p?N?'@:Preprocessing2U
Iterator::Model::ParallelMapV2?|?5^???!?a~??'@)?|?5^???1?a~??'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?S㥛İ?!?6E??@)?S㥛İ?1?6E??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???B?i??!<? )?4@)K?46??1Q?PT@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???Q???!P̃E_,@)??C?l???1b?[x?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*M??St$??!??'ϩ_@)M??St$??1??'ϩ_@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9/i??Y@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	u??y????]vn1ڵ?A?c?]K??!bX9????	!       "	!       *	!       2$	+uu2!???M??lE??;M?O??!Ԛ?????:	!       B	!       J$	R?c?n???m??T؀?	?^)ː?!	?c???R	!       Z$	R?c?n???m??T؀?	?^)ː?!	?c???JCPU_ONLYY/i??Y@b Y      Y@q_???yiA@"?
both?Your program is POTENTIALLY input-bound because 45.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.824% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 