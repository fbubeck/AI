?($	???\??!7????
ףp=
??!(~??k	??$	h0v?? @E??|??@?y?=?@!}A_??2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q?????^)?Ǻ?A?H?}8??Yw??/ݴ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C?????Q???A,e?X??Y??(\?¥?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)?????B?i??A??q????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??St$?????e??a??AZd;?O??Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B>?٬???A???Q???Y?f??j+??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]m???{?????K7???A?}8gD??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?$??+??	h??A?? ?rh??Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4??9??m4???AM?O???Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0???߾?3??A??????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	:#J{?/?????z6??A+?????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
???&???٬?\m??AgDio????Y<?R?!???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(~??k	??(~??k	??A#J{?/L??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????ׁ???]K?=??A???H??Y??y?)??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???B?i?????&S??A>yX?5???Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G????|?5^???A??K7?A??Y?"??~j??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M??St$??Y?8??m??A?? ???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ????|a2U??A?H?}??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g??Qk?w????Avq?-??Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}????bX9????Ao??ʡ??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w????L7?A`???A???????Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
??2U0*???A|a2U0??Y?:pΈҞ?*	     H?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?lV}????!
?&.s|B@)??@?????1??M?RA@:Preprocessing2F
Iterator::Model??^??!1?J?\?A@)?K7?A`??19????4@:Preprocessing2U
Iterator::Model::ParallelMapV2x??#????!U?[??,@)x??#????1U?[??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ZB>????!??ڦQ)P@)?ŏ1w??1
6?q'?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceg??j+???!??)?@)g??j+???1??)?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateOjM???!???؊?+@)8gDio???1>D???h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?z6?>??!?0?L?2@)??ǘ????1?ʎf?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?W[?????!????@)?W[?????1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9L????6@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	2?A????r?	?y???2U0*???!(~??k	??	!       "	!       *	!       2$	q)?r)????;?=?\??|a2U0??!??K7?A??:	!       B	!       J$	XI?9??????R????X?? ??!w??/ݴ?R	!       Z$	XI?9??????R????X?? ??!w??/ݴ?JCPU_ONLYYL????6@b Y      Y@qƨ?>?@@"?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?33.541% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 