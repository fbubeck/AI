?($	_?L?????g????y?)??!?(\?????$	??j>??@Xp?r	?@?aB~ @!?jq?w-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$H?z?G????_?L??A???~?:??Y\ A?c̭?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(?????k	??g??AO@a????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)??w??/???A??K7?A??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A??T㥛? ??Ar?鷯??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?B?i?q??$???????A????????YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г??鷯????AH?z?G??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???镲???ݓ??Z??A???3???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4???-?????A?q?????Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|?5^???	h"lx??A?ڊ?e???YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??_?L??`vOj??A4??@????Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?$??C??S?!?uq??A^K?=???Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ??>?٬?\??A?N@a???Y'?Wʢ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[????<???]K?=??A?ܵ?|???YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q???$??C??Ash??|???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s???#J{?/L??A???o_??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?y?):???????<,??A鷯???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7??"lxz?,??AW?/?'??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w?????C??????A?3??7??YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?L?J?????ʡE??A?ʡE????Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(\??????rh??|??AгY?????Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)??46<?R??A_?Q???Ysh??|???*	?????i?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO@a????!????u	A@)?;Nё\??1"T?v"??@:Preprocessing2F
Iterator::Model?St$????!?L?B@)]?C?????1?!j5@:Preprocessing2U
Iterator::Model::ParallelMapV2???N@??!????/X-@)???N@??1????/X-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF??_???!$????O@)??^??1<dFo??%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate.?!??u??!e>?
?I,@)??ǘ????1??=;2?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?|a2U??!Ч?ّ@)?|a2U??1Ч?ّ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?/L?
??!??????2@)46<?R??1?}?Ń.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*HP?s??!?z]K&@)HP?s??1?z]K&@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no957R?,?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	hO?L ?????!??????_?L??!?rh??|??	!       "	!       *	!       2$	i?^4???s????_?Q???!鷯???:	!       B	!       J$	?Y ????0?"]?e???j+??ݓ?!\ A?c̭?R	!       Z$	?Y ????0?"]?e???j+??ݓ?!\ A?c̭?JCPU_ONLYY57R?,?@b Y      Y@qJ2١??@"?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.5845% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 