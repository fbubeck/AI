?($	\?,?;???0?n[???ŏ1w-!??!?[ A?c??$	?yet?@%T/FB??7Ğ????!L?-?E%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$鷯??????@?????A?HP???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)???? ?o_???A?=yX?5??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)????b?=y??Aq?-???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾???9??m4???A?.n????YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM????H?}??A?e?c]???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;????X?? ??A$???????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R???`??"????A?&?W??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????ׁsF??A?.n????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A?????Pk?w??A?,C????Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?/?'???Y??ڊ??AA??ǘ???YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?~?:p????^)????A(~??k	??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ???؁sF????A:??H???Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c??A??ǘ???A????߾??Yf??a?֤?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/????A?f???A?]K?=??Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+???f??j+??A??A?f??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO???ǘ?????Aڬ?\m???YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??s????9??v????AjM????Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞???(~??k	??A??d?`T??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Pk?w??HP?s??A????<,??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??Nё\?C??Ad?]K???YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-!??(??y??A??镲??Y46<???*	fffff"?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR'??????!~?
F?B@)?c]?F??1?|UG?2@@:Preprocessing2F
Iterator::Model?ͪ??V??!???O?@@)???V?/??1cmT??h3@:Preprocessing2U
Iterator::Model::ParallelMapV2$???~???!???/?K+@)$???~???1???/?K+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%??C???!,,pX?P@)?~?:pθ?1?{8)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??H.?!??!??a?lx@)??H.?!??1??a?lx@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatez6?>W??!?|[?+@)x$(~???1D?Z?I?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?
F%u??!0]?72@)z6?>W[??1?"?gՎ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?<,Ԛ???!??P?O?@)?<,Ԛ???1??P?O?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???}N|@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	!?A7?????L??b??(??y??!?^)????	!       "	!       *	!       2$	?*?-????3?R?????镲??!????߾??:	!       B	!       J$	???xv????<?3'˄?S?!?uq??!??_vO??R	!       Z$	???xv????<?3'˄?S?!?uq??!??_vO??JCPU_ONLYY???}N|@b Y      Y@q.@ יwA@"?
both?Your program is POTENTIALLY input-bound because 53.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.9344% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 